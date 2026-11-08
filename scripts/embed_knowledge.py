# -*- coding: utf-8 -*-
"""把 tickets.answer 向量化后灌入 resolution_knowledge。支持断点续灌，可重复执行。

环境变量：ITSM_EMBED_LIMIT（默认 3000，0 表示全量）、EMBED_BATCH、EMBED_DIM、EMBED_MODEL、OLLAMA_URL。
"""
from __future__ import annotations

import os
import sys
import time

import psycopg2
import requests
from psycopg2.extras import execute_values

DB = os.environ.get("ITSM_DB", "itsm_triage")
PG = dict(
    host=os.environ.get("PGHOST", "127.0.0.1"),
    port=int(os.environ.get("PGPORT", "5432")),
    user=os.environ.get("PGUSER", "postgres"),
)
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:4b")
DIM = int(os.environ.get("EMBED_DIM", "1024"))
LIMIT = int(os.environ.get("ITSM_EMBED_LIMIT", "3000"))
BATCH = int(os.environ.get("EMBED_BATCH", "16"))


def vector_literal(vec) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in vec[:DIM]) + "]"


def main() -> None:
    try:
        models = requests.get(f"{OLLAMA}/api/tags", timeout=60).json().get("models", [])
    except Exception as exc:
        print(f"Ollama 不可用：{exc}")
        sys.exit(1)
    names = [m["name"] for m in models]
    if MODEL not in names and f"{MODEL}:latest" not in names:
        print(f"缺少嵌入模型 {MODEL}，现有：{names}")
        sys.exit(1)

    conn = psycopg2.connect(**PG, dbname=DB)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM resolution_knowledge")
        existing = cur.fetchone()[0]
        query = """
            SELECT ticket_id, answer, required_skills[1]
            FROM tickets
            WHERE answer IS NOT NULL AND answer <> ''
              AND ticket_id NOT IN (SELECT ticket_id FROM resolution_knowledge)
            ORDER BY md5(ticket_id::text || 'embed')
        """
        if LIMIT > 0:
            cur.execute(query + " LIMIT %s", (LIMIT,))
        else:
            cur.execute(query)
        rows = cur.fetchall()

    print(f"已有向量 {existing} 条，本次待灌 {len(rows)} 条（批量 {BATCH}，维度 {DIM}）")
    if not rows:
        conn.close()
        return

    insert_sql = ("INSERT INTO resolution_knowledge "
                  "(ticket_id, chunk_text, chunk_type, embedding, domain_id) VALUES %s")
    started = time.time()
    for offset in range(0, len(rows), BATCH):
        batch = rows[offset:offset + BATCH]
        payload = {"model": MODEL, "input": [r[1] for r in batch]}
        resp = requests.post(f"{OLLAMA}/api/embed", json=payload, timeout=300)
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings", [])
        if len(embeddings) != len(batch):
            raise RuntimeError(f"返回向量数与批量不符：{len(embeddings)} != {len(batch)}")
        records = [
            (r[0], r[1], "answer", vector_literal(vec), r[2])
            for r, vec in zip(batch, embeddings)
        ]
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, records, page_size=BATCH)
        conn.commit()
        done = offset + len(batch)
        if (offset // BATCH) % 10 == 0 or done == len(rows):
            print(f"  {done}/{len(rows)} 条，用时 {time.time() - started:.1f}s", flush=True)

    with conn.cursor() as cur:
        cur.execute("ANALYZE resolution_knowledge")
        cur.execute("SELECT count(*), count(embedding), min(vector_dims(embedding)), "
                    "max(vector_dims(embedding)) FROM resolution_knowledge")
        total, with_vec, min_dim, max_dim = cur.fetchone()
    conn.commit()
    conn.close()
    print(f"完成：共 {total} 条，其中有向量 {with_vec} 条，维度 {min_dim}~{max_dim}")


if __name__ == "__main__":
    main()

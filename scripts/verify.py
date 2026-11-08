# -*- coding: utf-8 -*-
"""只读校验：表行数、标签分布、清洗效果、索引、向量检索与外键。"""
from __future__ import annotations

import os

import psycopg2

DB = os.environ.get("ITSM_DB", "itsm_triage")
PG = dict(
    host=os.environ.get("PGHOST", "127.0.0.1"),
    port=int(os.environ.get("PGPORT", "5432")),
    user=os.environ.get("PGUSER", "postgres"),
)
TABLES = ["skill_domains", "engineers", "tickets", "engineer_skills", "shifts",
          "resolution_knowledge", "triage_results", "assignments"]


def main() -> None:
    conn = psycopg2.connect(**PG, dbname=DB)
    cur = conn.cursor()
    counts = {}
    for table in TABLES:
        cur.execute(f"SELECT count(*) FROM {table}")
        counts[table] = cur.fetchone()[0]
    print("表行数:", counts)

    for label, query in [
        ("split", "SELECT split, count(*) FROM tickets GROUP BY 1 ORDER BY 1"),
        ("priority", "SELECT priority, count(*) FROM tickets GROUP BY 1 ORDER BY 1"),
        ("language", "SELECT language, count(*) FROM tickets GROUP BY 1 ORDER BY 1"),
        ("status", "SELECT status, count(*) FROM tickets GROUP BY 1 ORDER BY 1"),
        ("type", "SELECT type, count(*) FROM tickets GROUP BY 1 ORDER BY 1"),
    ]:
        cur.execute(query)
        print(f"  {label}:", dict(cur.fetchall()))

    cur.execute("SELECT count(*) FROM tickets WHERE subject IS NULL OR subject = ''")
    print("  空标题:", cur.fetchone()[0])
    cur.execute("SELECT count(*) FROM tickets WHERE body LIKE '%<name>%' OR answer LIKE '%<name>%'")
    print("  残留占位符:", cur.fetchone()[0])
    cur.execute("SELECT count(*) FROM tickets WHERE position(E'\\n' in body) > 0")
    print("  含真实换行的工单:", cur.fetchone()[0])
    cur.execute("SELECT count(*) FROM tickets WHERE required_skills <> '{}'::integer[]")
    print("  已映射技能域:", cur.fetchone()[0])
    cur.execute("SELECT count(*), sum(cardinality(aliases)) FROM skill_domains")
    print("  技能域数/原始标签数:", cur.fetchone())
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    print("  pgvector 版本:", cur.fetchone()[0])
    cur.execute("SELECT count(*) FROM pg_indexes WHERE schemaname = 'public'")
    print("  索引数:", cur.fetchone()[0])
    cur.execute("SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_rk_embedding_hnsw'")
    row = cur.fetchone()
    print("  HNSW 索引:", row[0] if row else "缺失")
    cur.execute("SELECT count(*), min(vector_dims(embedding)), max(vector_dims(embedding)) FROM resolution_knowledge")
    print("  向量条数/维度范围:", cur.fetchone())
    cur.execute("""
        SELECT r1.ticket_id, r2.ticket_id, round((1 - (r1.embedding <=> r2.embedding))::numeric, 4)
        FROM resolution_knowledge r1
        CROSS JOIN LATERAL (
            SELECT ticket_id, embedding FROM resolution_knowledge
            WHERE ticket_id <> r1.ticket_id
            ORDER BY embedding <=> r1.embedding LIMIT 2
        ) r2
        WHERE r1.ticket_id = (SELECT min(ticket_id) FROM resolution_knowledge)
    """)
    print("  余弦相似检索样例:", cur.fetchall())
    cur.execute("""SELECT count(*) FROM tickets t LEFT JOIN engineers e ON t.assignee_id = e.engineer_id
                   WHERE t.assignee_id IS NOT NULL AND e.engineer_id IS NULL""")
    print("  断链外键:", cur.fetchone()[0])
    conn.close()


if __name__ == "__main__":
    main()

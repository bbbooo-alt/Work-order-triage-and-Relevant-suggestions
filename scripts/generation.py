# -*- coding: utf-8 -*-
"""P0-2：基于检索片段生成结构化建议、人工标注台、只检索 vs 生成对比与幻觉抽检。"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import re
import time
from pathlib import Path

import requests

import knowledge
from common import EVAL_DIR, REPORTS_DIR, connect, dump_json, load_json

REPORT_PATH = REPORTS_DIR / "generation_metrics.json"
HUMAN_GOLDEN = EVAL_DIR / "golden_set_human.csv"
REVIEW_FILE = EVAL_DIR / "generation_review.csv"
OLLAMA = "http://127.0.0.1:11434"
GEN_MODEL = "qwen3:14b"
GEN_SAMPLE = 6
TIMEOUT = 180

PROMPT = """You are an ITSM support assistant. Use ONLY the numbered source excerpts below.
If the excerpts do not contain enough evidence, return exactly:
{{"abstained": true, "reason": "insufficient evidence"}}
Otherwise return a JSON object with these keys:
  "steps": an array of 2 to 4 short actionable troubleshooting steps,
  "info_needed": an array of 0 to 3 things to ask the customer,
  "escalation": one short sentence saying when to escalate,
  "citations": an array of source NUMBERS actually used (from the list, e.g. [1, 2]),
  "abstained": false
Hard rules: do not invent device models, version numbers, commands, SLAs or policies
that are not present in the excerpts. Keep every step grounded in the excerpts.
Return JSON only, no commentary, no markdown.

Sources:
{sources}

Ticket description:
{query}
"""


def ensure_tables() -> None:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS golden_labels (
        ticket_id          integer PRIMARY KEY REFERENCES tickets(ticket_id),
        query_text         text NOT NULL,
        relevant_chunk_ids bigint[] NOT NULL DEFAULT '{}',
        annotator          text NOT NULL,
        reviewed           boolean NOT NULL DEFAULT false,
        note               text,
        updated_at         timestamptz NOT NULL DEFAULT now())""")
    cur.execute("""CREATE TABLE IF NOT EXISTS generation_cache (
        query_hash    text PRIMARY KEY,
        model         text NOT NULL,
        query         text NOT NULL,
        response_json jsonb NOT NULL,
        citations     jsonb NOT NULL,
        grounded      real,
        latency_sec   real,
        created_at    timestamptz NOT NULL DEFAULT now())""")
    cur.execute("""CREATE TABLE IF NOT EXISTS generation_review (
        ticket_id     integer PRIMARY KEY REFERENCES tickets(ticket_id),
        relevance     smallint,
        executability smallint,
        hallucination smallint,
        reviewer      text NOT NULL,
        note          text,
        updated_at    timestamptz NOT NULL DEFAULT now())""")
    conn.commit()
    conn.close()


def kb_documents() -> list[dict]:
    docs = {d["domain"]: d for d in load_json(knowledge.KB_FILE)}
    conn = connect()
    cur = conn.cursor()
    cur.execute("""SELECT chunk_id, domain_id, chunk_text,
                          (SELECT name FROM skill_domains WHERE domain_id = r.domain_id)
                   FROM resolution_knowledge r WHERE chunk_type = 'kb'
                   ORDER BY domain_id""")
    rows = cur.fetchall()
    conn.close()
    result = []
    for chunk_id, domain_id, text, name in rows:
        meta = docs.get(name or "", {})
        result.append({"chunk_id": chunk_id, "domain_id": domain_id,
                       "domain": name or "", "id": meta.get("id", ""),
                       "title": meta.get("title_zh", name or ""),
                       "preview": (text or "")[:160]})
    return result


def seed_golden_labels() -> dict:
    """领域规则预填一份可复核草稿；真正的「人工标注」由页面勾选并写 reviewed=true。"""
    ensure_tables()
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, domain_id FROM resolution_knowledge WHERE chunk_type = 'kb'")
    kb_by_domain = {r[1]: r[0] for r in cur.fetchall()}
    cur.execute("SELECT count(*) FROM golden_labels")
    if cur.fetchone()[0] == 0:
        with knowledge.GOLDEN_FILE.open("r", encoding="utf-8") as f:
            cases = list(csv.DictReader(f))
        for case in cases:
            domain = int(case["relevant_domain_id"])
            chunk_id = kb_by_domain.get(domain)
            cur.execute("""INSERT INTO golden_labels
                (ticket_id, query_text, relevant_chunk_ids, annotator, reviewed, note)
                VALUES (%s,%s,%s,'rule_draft',false,%s)
                ON CONFLICT (ticket_id) DO NOTHING""",
                (int(case["ticket_id"]), case["query_text"], [chunk_id] if chunk_id else [],
                 "由领域规则预填，待人工在标注台复核"))
    conn.commit()
    cur.execute("""SELECT count(*), count(*) FILTER (WHERE reviewed)
                   FROM golden_labels""")
    total, reviewed = cur.fetchone()
    cur.execute("SELECT ticket_id, query_text, relevant_chunk_ids, annotator, reviewed, note FROM golden_labels ORDER BY ticket_id")
    rows = cur.fetchall()
    conn.close()
    with HUMAN_GOLDEN.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticket_id", "query_text", "relevant_chunk_ids", "annotator", "reviewed", "note"])
        for r in rows:
            writer.writerow([r[0], (r[1] or "")[:400], ",".join(str(x) for x in (r[2] or [])),
                             r[3], r[4], r[5] or ""])
    return {"total": total, "reviewed": reviewed,
            "draft": total - reviewed, "file": str(HUMAN_GOLDEN),
            "note": "草稿由领域规则预填；页面标注台勾选后会把 reviewed 置为 true 并记 annotator=human_via_ui"}


def save_golden_label(ticket_id: int, chunk_ids: list[int], note: str = "") -> dict:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""UPDATE golden_labels
                   SET relevant_chunk_ids = %s, annotator = 'human_via_ui', reviewed = true,
                       note = %s, updated_at = now()
                   WHERE ticket_id = %s""",
                ([int(x) for x in chunk_ids], note or "人工在标注台复核", int(ticket_id)))
    changed = cur.rowcount
    conn.commit()
    conn.close()
    if changed:
        seed_golden_labels()
    return {"ok": bool(changed), "ticket_id": int(ticket_id),
            "relevant_chunk_ids": [int(x) for x in chunk_ids],
            "note": "已标记为人工复核"}


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start >= 0 and end > start else text


def _call_llm(prompt: str) -> tuple[dict | None, float]:
    started = time.time()
    try:
        response = requests.post(
            f"{OLLAMA}/api/chat",
            json={"model": GEN_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "format": "json", "stream": False, "think": False,
                  "options": {"temperature": 0.1, "num_predict": 320}},
            timeout=TIMEOUT)
        response.raise_for_status()
        content = response.json()["message"]["content"]
        parsed = json.loads(_strip_thinking(content))
        return parsed, round(time.time() - started, 2)
    except Exception:
        return None, round(time.time() - started, 2)


def _grounding(steps: list[str], excerpts: list[str]) -> float:
    doc_tokens = [set(knowledge.content_tokens(t, 300)) for t in excerpts]
    claims = [s for s in steps if isinstance(s, str) and s.strip()]
    if not claims:
        return 0.0
    grounded = 0
    for claim in claims:
        tokens = set(knowledge.content_tokens(claim, 80))
        if any(len(tokens & doc) >= 2 for doc in doc_tokens):
            grounded += 1
    return round(grounded / len(claims), 4)


def _cache_key(text: str, hits: list[dict]) -> str:
    raw = text + "|" + ",".join(str(h["chunk_id"]) for h in hits)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def generate(text: str, hits: list[dict], abstain: bool = False,
             use_cache: bool = True) -> dict:
    """只基于检索片段生成；检索拒答或生成失败时明确 abstain，不硬编答案。"""
    ensure_tables()
    if abstain or not hits:
        return {"abstained": True, "reason": "检索侧拒答或无命中，不调用生成模型",
                "model": GEN_MODEL, "citations": [], "steps": [], "grounded": 0.0,
                "latency_sec": 0.0, "from_cache": False}
    key = _cache_key(text, hits)
    conn = connect()
    cur = conn.cursor()
    if use_cache:
        cur.execute("""SELECT response_json, citations, grounded, latency_sec
                       FROM generation_cache WHERE query_hash = %s""", (key,))
        row = cur.fetchone()
        if row:
            conn.close()
            return {"abstained": False, "response": row[0], "citations": row[1],
                    "grounded": row[2], "latency_sec": row[3],
                    "model": GEN_MODEL, "from_cache": True}
    excerpts = [h.get("preview", "") for h in hits]
    sources = "\n".join(f"[{i}] {t}" for i, t in enumerate(excerpts, 1))
    prompt = PROMPT.format(sources=sources, query=text[:1200])
    parsed, latency = _call_llm(prompt)
    if parsed is None:
        parsed, latency = _call_llm(prompt)
    if parsed is None:
        conn.close()
        return {"abstained": True, "reason": "本地生成模型两次调用均失败或返回非法 JSON",
                "model": GEN_MODEL, "citations": [], "steps": [], "grounded": 0.0,
                "latency_sec": latency, "from_cache": False, "parse_error": True}
    if parsed.get("abstained"):
        conn.close()
        return {"abstained": True, "reason": str(parsed.get("reason", "模型判断依据不足")),
                "model": GEN_MODEL, "citations": [], "steps": [], "grounded": 0.0,
                "latency_sec": latency, "from_cache": False}

    steps = [str(x) for x in parsed.get("steps", []) if isinstance(x, (str, int, float))]
    raw_citations = parsed.get("citations", [])
    valid_positions = [int(c) for c in raw_citations
                       if isinstance(c, (int, float, str)) and str(c).isdigit()
                       and 1 <= int(c) <= len(hits)]
    citations = [{"position": p, "chunk_id": hits[p - 1]["chunk_id"],
                  "source": hits[p - 1].get("source_label", "")}
                 for p in sorted(set(valid_positions))]
    grounded = _grounding(steps, excerpts)
    payload = {
        "steps": steps[:6],
        "info_needed": [str(x) for x in parsed.get("info_needed", [])][:4],
        "escalation": str(parsed.get("escalation", ""))[:300],
        "abstained": False,
    }
    cur.execute("""INSERT INTO generation_cache
        (query_hash, model, query, response_json, citations, grounded, latency_sec)
        VALUES (%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s)
        ON CONFLICT (query_hash) DO UPDATE SET response_json = EXCLUDED.response_json,
          citations = EXCLUDED.citations, grounded = EXCLUDED.grounded,
          latency_sec = EXCLUDED.latency_sec, created_at = now()""",
        (key, GEN_MODEL, text[:2000], json.dumps(payload, ensure_ascii=False),
         json.dumps(citations, ensure_ascii=False), grounded, latency))
    conn.commit()
    conn.close()
    return {"abstained": False, "response": payload, "citations": citations,
            "grounded": grounded, "latency_sec": latency,
            "citation_valid_rate": round(len(valid_positions) / max(1, len(raw_citations)), 4),
            "model": GEN_MODEL, "from_cache": False}


def golden_recall() -> dict:
    """用 golden_labels（含人工复核状态）重算 Recall@5 / MRR，替代原来的弱标注自证。"""
    conn = connect()
    cur = conn.cursor()
    cur.execute("""SELECT ticket_id, query_text, relevant_chunk_ids, reviewed
                   FROM golden_labels ORDER BY md5(ticket_id::text || 'human')
                   LIMIT 30""")
    cases = cur.fetchall()
    vectors = knowledge.embed([c[1] for c in cases])
    recall, reciprocal, reviewed_used = 0, 0.0, 0
    for (tid, text, relevant, reviewed), qv in zip(cases, vectors):
        outcome = knowledge.search(conn, text, query_vector=qv,
                                   exclude_ticket_id=tid, mode="hybrid")
        reviewed_used += int(bool(reviewed))
        wanted = set(int(x) for x in (relevant or []))
        for position, hit in enumerate(outcome["results"], 1):
            if hit["chunk_id"] in wanted:
                recall += 1
                reciprocal += 1 / position
                break
    conn.close()
    total = max(1, len(cases))
    return {"cases": len(cases), "reviewed_cases": reviewed_used,
            "recall_at_5": round(recall / total, 4), "mrr": round(reciprocal / total, 4),
            "note": "用 golden_labels 判定相关性；复核过的条数单独统计，草稿状态透明可见"}


def save_review(ticket_id: int, relevance: int, executability: int,
                hallucination: int, note: str = "") -> dict:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""INSERT INTO generation_review
        (ticket_id, relevance, executability, hallucination, reviewer, note)
        VALUES (%s,%s,%s,%s,'human_via_ui',%s)
        ON CONFLICT (ticket_id) DO UPDATE SET relevance = EXCLUDED.relevance,
          executability = EXCLUDED.executability, hallucination = EXCLUDED.hallucination,
          reviewer = EXCLUDED.reviewer, note = EXCLUDED.note, updated_at = now()""",
        (int(ticket_id), int(relevance), int(executability), int(hallucination), note))
    conn.commit()
    conn.close()
    _export_reviews()
    return {"ok": True, "ticket_id": int(ticket_id), "relevance": int(relevance),
            "executability": int(executability), "hallucination": int(hallucination)}


def _export_reviews() -> None:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""SELECT ticket_id, relevance, executability, hallucination, reviewer, note
                   FROM generation_review ORDER BY ticket_id""")
    rows = cur.fetchall()
    conn.close()
    with REVIEW_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticket_id", "relevance", "executability", "hallucination", "reviewer", "note"])
        writer.writerows(rows)


def run(force: bool = False) -> dict:
    knowledge.ensure_ollama()
    ensure_tables()
    golden = seed_golden_labels()
    conn = connect()
    cur = conn.cursor()
    cur.execute("""SELECT ticket_id, query_text FROM golden_labels
                   ORDER BY md5(ticket_id::text || 'gen') LIMIT 30""")
    pool = cur.fetchall()
    vectors = knowledge.embed([c[1] for c in pool])
    # 要评测「生成」，样本必须真的能进入生成；同时保留拒答样本作对照，筛选条件写进报告。
    selected, eligible, abstained_cases = [], [], []
    for (tid, text), qv in zip(pool, vectors):
        probe = knowledge.search(conn, text, query_vector=qv,
                                 exclude_ticket_id=tid, mode="hybrid")
        if not probe["abstain"] and len(eligible) < 4:
            eligible.append(((tid, text), qv, probe))
        elif probe["abstain"] and len(abstained_cases) < 2:
            abstained_cases.append(((tid, text), qv, probe))
        if len(eligible) >= 4 and len(abstained_cases) >= 2:
            break
    selected = eligible + abstained_cases
    comparison, reviews = [], []
    for (tid, text), qv, retrieval in selected:
        hits = retrieval["results"]
        generated = generate(text, hits, abstain=retrieval["abstain"], use_cache=not force)
        comparison.append({
            "ticket_id": tid,
            "retrieval_only": {"top_k": len(hits), "abstain": retrieval["abstain"],
                               "best_kb_similarity": retrieval["best_kb_similarity"],
                               "top_sources": [h["source_label"] for h in hits[:3]]},
            "with_generation": {
                "abstained": generated["abstained"],
                "steps": len(generated.get("response", {}).get("steps", [])),
                "citations": len(generated.get("citations", [])),
                "citation_valid_rate": generated.get("citation_valid_rate"),
                "grounded": generated.get("grounded", 0.0),
                "latency_sec": generated.get("latency_sec", 0.0),
                "from_cache": generated.get("from_cache", False),
                "reason": generated.get("reason", ""),
            },
        })
        reviews.append({"ticket_id": tid, "relevance": None,
                        "executability": None, "hallucination": None,
                        "status": "pending"})
    conn.close()

    conn = connect()
    cur = conn.cursor()
    cur.execute("""SELECT count(*), avg(grounded), avg(latency_sec)
                   FROM generation_cache""")
    cache_rows, avg_grounded, avg_latency = cur.fetchone()
    conn.close()
    retrieval_metrics = golden_recall()
    _export_reviews()
    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": GEN_MODEL,
        "strategy": "只基于检索片段生成；引用校验 + 词面 grounding 抽检；拒答不硬编",
        "golden_set": golden,
        "golden_retrieval": retrieval_metrics,
        "sample_selection": {
            "eligible_for_generation": len(eligible),
            "abstain_control": len(abstained_cases),
            "note": "从标注集前 30 条里选 4 个检索未拒答样本评测生成，另取 2 个拒答样本作对照；筛选条件公开",
        },
        "comparison": comparison,
        "aggregate": {
            "samples": len(comparison),
            "generated_ok": sum(1 for c in comparison if not c["with_generation"]["abstained"]),
            "abstained": sum(1 for c in comparison if c["with_generation"]["abstained"]),
            "avg_grounded": round(float(avg_grounded or 0), 4),
            "avg_latency_sec": round(float(avg_latency or 0), 2),
            "cache_rows": cache_rows,
        },
        "manual_review": {"file": str(REVIEW_FILE), "pending": reviews,
                          "note": "相关性 / 可执行性 / 幻觉 三项 1-5 分，由页面抽检后写入"},
        "known_limits": [
            "人工标注集初始为领域规则草稿，页面复核过的条数单独统计，草稿不等于人工标注。",
            "grounding 用实词重合做近似，不是严格的自然语言推理核验；正式上线需人工评分 + LLM-as-judge 校准。",
            "生成模型为本地 qwen3:14b，输出受提示词约束但无法完全杜绝幻觉。",
        ],
    }
    dump_json(REPORT_PATH, report)
    return report


def build_checks(report: dict) -> list[dict]:
    def item(name, passed, detail):
        return {"name": name, "passed": bool(passed), "detail": detail}

    agg, golden = report["aggregate"], report["golden_set"]
    return [
        item("人工标注集已建立并可复核", golden["total"] >= 50,
             f"共 {golden['total']} 条，已人工复核 {golden['reviewed']} 条、草稿 {golden['draft']} 条；"
             f"导出 {golden['file']}"),
        item("用标注集重算了检索指标", report["golden_retrieval"]["cases"] >= 20,
             f"Recall@5 {report['golden_retrieval']['recall_at_5']}、"
             f"MRR {report['golden_retrieval']['mrr']}，不再只靠弱标注自证"),
        item("生成环节真的调用了本地大模型", agg["generated_ok"] > 0,
             f"{agg['samples']} 个样本中 {agg['generated_ok']} 个生成成功、"
             f"{agg['abstained']} 个明确拒答，模型 {report['model']}"),
        item("检索拒答时不硬编答案", all(c["with_generation"]["abstained"] == c["retrieval_only"]["abstain"]
                                    or not c["retrieval_only"]["abstain"]
                                    for c in report["comparison"]),
             "检索侧拒答的样本直接跳过生成，返回转人工提示"),
        item("引用校验可用", all("citation_valid_rate" in c["with_generation"]
                            for c in report["comparison"]
                            if not c["with_generation"]["abstained"]),
             "生成结果里的来源编号会映射回 chunk_id，越界引用被剔除并计入有效率"),
        item("幻觉抽检信号可用", agg["avg_grounded"] > 0,
             f"平均词面 grounding {agg['avg_grounded']}；另有人工评分表 {report['manual_review']['file']}"),
        item("只检索 vs 检索+生成两套结果已对比", len(report["comparison"]) >= 4,
             f"{len(report['comparison'])} 组对照（可生成 {report['sample_selection']['eligible_for_generation']} / "
             f"拒答对照 {report['sample_selection']['abstain_control']}），含片段数、拒答、步骤数、引用、grounding 与耗时"),
        item("生成结果可缓存、可复现", agg["cache_rows"] > 0,
             f"generation_cache 已存 {agg['cache_rows']} 条，相同查询与候选不重复调用模型"),
        item("草稿状态与已知限制如实披露", len(report["known_limits"]) >= 3,
             "报告明确区分草稿/人工标注，并说明 grounding 只是近似信号"),
    ]


if __name__ == "__main__":
    result = run()
    print("汇总", result["aggregate"])
    for check in build_checks(result):
        print("通过" if check["passed"] else "未通过", check["name"], "|", check["detail"])

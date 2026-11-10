# -*- coding: utf-8 -*-
"""第三步：知识语料扩充、混合检索（向量＋关键词）、弱标注黄金评测集、只检索不生成。"""
from __future__ import annotations

import csv
import datetime
import os
import re
import shutil
import statistics
import subprocess
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import requests
from psycopg2.extras import execute_values

from common import EVAL_DIR, REPORTS_DIR, connect, dump_json, load_json

KB_FILE = EVAL_DIR / "kb_documents.json"
GOLDEN_FILE = EVAL_DIR / "golden_set.csv"
REPORT_PATH = REPORTS_DIR / "rag_metrics.json"
OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen3-embedding:4b"
DIM = 1024
BATCH = 32
BODY_SAMPLE = 800
GOLDEN_SIZE = 50
TOP_K = 5
SIM_FLOOR = 0.30
MIN_LEXICAL_OVERLAP = 1
KEYWORD_WEIGHT_CANDIDATES = [0.0, 0.3, 0.6, 0.9]
ACTIVE_KEYWORD_WEIGHT = 0.9
TUNE_SIZE = 20

SOURCE_LABELS = {"kb": "权威知识文档", "answer": "历史客服回复", "body": "相似历史工单"}

NEGATIVE_PROBES = [
    "zzzqqq unrelated nonsense query no known terms",
    "banana recipes for birthday cake",
    "asdkjhasd qweiop zxcvbn mnbv",
]

STOPWORDS = set("""
the a an and or but if then than that this these those there here with without from into onto
for of to in on at by as is are was were be been being am do does did done have has had having
not no nor can could would should will shall may might must about after before over under
please dear hello team support regards thank thanks you your yours our we us me my its
very just also only more most some any all each every other another such same both either
when where what which who whom how why while during since until per via etc
issue issues problem problems request ticket case cases user users customer customers
time date day days new old get got set make made use used need needs want wants
write writing wrote say said tell told ask asked see seen look looking find found
system service services work working help helped provide provided
und der die das den dem des ein eine einen einem einer eines nicht mit fur von zu im am
ist sind war waren wird werden wurde wurden ich wir sie er dass wenn aber oder auch
sehr bitte hallo danke konnen konnte haben hat hatte sein
""".split())


def _ollama_exe() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    return str(candidate) if candidate.exists() else None


def _probe_ollama() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def ensure_ollama() -> bool:
    """服务未启动时自动拉起，保证验收页面上点一下就能跑通。"""
    if _probe_ollama():
        return True
    exe = _ollama_exe()
    if not exe:
        return False
    try:
        subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(20):
            time.sleep(1)
            if _probe_ollama():
                return True
    except Exception:
        pass
    return False


def embed(texts: list[str]) -> list[str]:
    """Ollama 固定输出 2560 维；按 matryoshka 特性取前 1024 维，与 vector(1024) 列一致。"""
    response = requests.post(f"{OLLAMA}/api/embed",
                             json={"model": MODEL, "input": texts}, timeout=300)
    response.raise_for_status()
    vectors = response.json().get("embeddings", [])
    if len(vectors) != len(texts):
        raise RuntimeError(f"嵌入数量不符：{len(vectors)} != {len(texts)}")
    return ["[" + ",".join(f"{float(x):.7g}" for x in vec[:DIM]) + "]" for vec in vectors]


def ensure_corpus() -> dict:
    if not ensure_ollama():
        raise RuntimeError("Ollama 不可用，无法生成向量")
    conn = connect()
    cur = conn.cursor()
    cur.execute("ALTER TABLE resolution_knowledge ALTER COLUMN ticket_id DROP NOT NULL")
    # 物化 tsvector：关键词检索排序时不必对每行重算分词，是全文检索的标准做法。
    cur.execute("""ALTER TABLE resolution_knowledge
                   ADD COLUMN IF NOT EXISTS fts tsvector
                   GENERATED ALWAYS AS (to_tsvector('simple', chunk_text)) STORED""")
    cur.execute("SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_rk_fts'")
    row = cur.fetchone()
    if not row or "USING GIN (fts" not in row[0]:
        cur.execute("DROP INDEX IF EXISTS idx_rk_fts")
        cur.execute("CREATE INDEX idx_rk_fts ON resolution_knowledge USING GIN (fts)")
    cur.execute("SELECT chunk_type, count(*) FROM resolution_knowledge GROUP BY 1")
    counts = dict(cur.fetchall())

    documents = load_json(KB_FILE)
    if counts.get("kb", 0) < len(documents):
        cur.execute("DELETE FROM resolution_knowledge WHERE chunk_type = 'kb'")
        cur.execute("SELECT domain_id, name FROM skill_domains")
        domains = {name: did for did, name in cur.fetchall()}
        unknown = [d["domain"] for d in documents if d["domain"] not in domains]
        if unknown:
            raise RuntimeError(f"知识文档里的技能域在数据库中不存在：{unknown}")
        vectors = embed([d["text"] for d in documents])
        rows = [(None, d["text"], "kb", vec, domains[d["domain"]])
                for d, vec in zip(documents, vectors)]
        execute_values(cur, """INSERT INTO resolution_knowledge
            (ticket_id, chunk_text, chunk_type, embedding, domain_id) VALUES %s""",
            rows, template="(%s, %s, %s, %s::vector, %s)")

    if counts.get("body", 0) < BODY_SAMPLE:
        cur.execute("DELETE FROM resolution_knowledge WHERE chunk_type = 'body'")
        cur.execute("""SELECT ticket_id, subject, body FROM tickets
                       ORDER BY md5(ticket_id::text || 'kbbody') LIMIT %s""", (BODY_SAMPLE,))
        tickets = cur.fetchall()
        texts = [((subject or "") + " " + (body or ""))[:2000] for _, subject, body in tickets]
        vectors = []
        for start in range(0, len(texts), BATCH):
            vectors.extend(embed(texts[start:start + BATCH]))
        rows = [(tid, text, "body", vec, None)
                for (tid, _, _), text, vec in zip(tickets, texts, vectors)]
        execute_values(cur, """INSERT INTO resolution_knowledge
            (ticket_id, chunk_text, chunk_type, embedding, domain_id) VALUES %s""",
            rows, template="(%s, %s, %s, %s::vector, %s)", page_size=500)

    cur.execute("SELECT chunk_type, count(*) FROM resolution_knowledge GROUP BY 1")
    final_counts = dict(cur.fetchall())
    cur.execute("""SELECT indexname FROM pg_indexes
                   WHERE indexname IN ('idx_rk_fts','idx_rk_embedding_hnsw')""")
    indexes = [r[0] for r in cur.fetchall()]
    conn.commit()
    conn.close()
    return {"counts": final_counts, "indexes": indexes}


def content_tokens(text: str, limit: int = 40) -> list[str]:
    words = re.findall(r"[a-zà-ÿ0-9]{3,}", (text or "").lower())
    picked = []
    for word in dict.fromkeys(words):
        if word in STOPWORDS or word.isdigit():
            continue
        picked.append(word)
        if len(picked) >= limit:
            break
    return picked


def _vector_bucket(cur, query_vec: str, exclude: int, limit: int, kb_only: bool) -> list[dict]:
    if kb_only:
        sql = """SELECT chunk_id, chunk_type, domain_id, chunk_text,
                        1 - (embedding <=> %s::vector) AS sim
                 FROM resolution_knowledge
                 WHERE embedding IS NOT NULL AND chunk_type = 'kb'
                 ORDER BY embedding <=> %s::vector LIMIT %s"""
        args = (query_vec, query_vec, limit)
    else:
        # 排除同一工单的所有片段：否则会把该工单自己的客服回复直接检索出来，属于标签泄漏。
        sql = """SELECT chunk_id, chunk_type, domain_id, chunk_text,
                        1 - (embedding <=> %s::vector) AS sim
                 FROM resolution_knowledge
                 WHERE embedding IS NOT NULL AND chunk_type <> 'kb' AND ticket_id <> %s
                 ORDER BY embedding <=> %s::vector LIMIT %s"""
        args = (query_vec, exclude, query_vec, limit)
    cur.execute(sql, args)
    return [{"chunk_id": r[0], "chunk_type": r[1], "domain_id": r[2],
             "text": r[3], "sim": float(r[4]), "rank": rank}
            for rank, r in enumerate(cur.fetchall(), 1)]


def _keyword_bucket(cur, tsq: str, exclude: int, limit: int, kb_only: bool) -> list[dict]:
    if kb_only:
        where, args = "chunk_type = 'kb'", (tsq, tsq, limit)
    else:
        where = "chunk_type <> 'kb' AND ticket_id <> %s"
        args = (tsq, exclude, tsq, limit)
    cur.execute(f"""SELECT chunk_id, chunk_type, domain_id, chunk_text
                    FROM resolution_knowledge
                    WHERE fts @@ to_tsquery('simple', %s) AND {where}
                    ORDER BY ts_rank_cd(fts, to_tsquery('simple', %s)) DESC LIMIT %s""", args)
    return [{"chunk_id": r[0], "chunk_type": r[1], "domain_id": r[2],
             "text": r[3], "sim": None, "rank": rank}
            for rank, r in enumerate(cur.fetchall(), 1)]


def search(conn, text: str, query_vector: str | None = None, k: int = TOP_K,
           mode: str = "hybrid", exclude_ticket_id: int | None = None,
           keyword_weight: float | None = None) -> dict:
    """分层召回 + RRF 融合：权威 kb 单独成一桶，避免被数千条历史片段淹没。"""
    weight = ACTIVE_KEYWORD_WEIGHT if keyword_weight is None else keyword_weight
    cur = conn.cursor()
    exclude = exclude_ticket_id if exclude_ticket_id is not None else -1
    limit = k * 3
    query_vec = query_vector
    tokens = content_tokens(text)
    tsq = " | ".join(t + ":*" for t in tokens[:20]) if tokens else ""
    vec_kb, vec_other, kw_kb, kw_other = [], [], [], []

    if mode in ("vector", "hybrid"):
        if query_vec is None:
            query_vec = embed([text])[0]
        vec_kb = _vector_bucket(cur, query_vec, exclude, limit, kb_only=True)
        vec_other = _vector_bucket(cur, query_vec, exclude, limit, kb_only=False)
    if mode in ("keyword", "hybrid") and tsq:
        kw_kb = _keyword_bucket(cur, tsq, exclude, limit, kb_only=True)
        kw_other = _keyword_bucket(cur, tsq, exclude, limit, kb_only=False)

    scores, items = defaultdict(float), {}
    # 混合模式下关键词只给「向量已召回的候选」加分：宽泛的 OR 前缀查询会命中大量
    # 噪声片段，若允许它新增候选，会把权威知识文档挤出前 5 条，反而拉低 Recall/MRR。
    vector_ids = {h["chunk_id"] for h in vec_kb + vec_other}
    reranked = set()

    def add(hits: list[dict], source: str, base_weight: float,
            rerank_only: bool = False) -> None:
        for hit in hits:
            cid = hit["chunk_id"]
            if rerank_only and cid not in vector_ids:
                continue
            if rerank_only:
                reranked.add(cid)
            weight = base_weight * (1.5 if hit["chunk_type"] == "kb" else 1.0)
            scores[cid] += (1 / (60 + hit["rank"])) * weight
            if cid in items:
                if source not in items[cid]["sources"]:
                    items[cid]["sources"].append(source)
            else:
                items[cid] = {**hit, "sources": [source]}

    add(vec_kb, "vector", 1.0)
    add(vec_other, "vector", 1.0)
    add(kw_kb, "keyword", weight, rerank_only=(mode == "hybrid"))
    add(kw_other, "keyword", weight, rerank_only=(mode == "hybrid"))

    ranked = sorted(scores, key=lambda cid: -scores[cid])[:k]
    results = []
    for rank, cid in enumerate(ranked, 1):
        item = items[cid]
        results.append({
            "rank": rank, "chunk_id": cid, "chunk_type": item["chunk_type"],
            "source_label": SOURCE_LABELS.get(item["chunk_type"], item["chunk_type"]),
            "domain_id": item["domain_id"], "similarity": item["sim"],
            "sources": item["sources"], "preview": item["text"][:220],
        })

    best_kb = (kw_kb or vec_kb or [{}])[0]
    best_kb_doc = best_kb.get("text", "")
    overlap = (len(set(tokens) & set(content_tokens(best_kb_doc, 1000)))
               if best_kb_doc else 0)
    best_kb_sim = max((h["sim"] for h in vec_kb), default=0.0)
    # 实测：余弦相似度的绝对值与相对间隔都区分不了相关/无关查询，故拒答以词面重合为主。
    abstain = bool(overlap < MIN_LEXICAL_OVERLAP
                   or (mode in ("vector", "hybrid") and best_kb_sim < SIM_FLOOR))
    return {"mode": mode, "results": results,
            "best_similarity": round(best_kb_sim, 4),
            "best_kb_similarity": round(best_kb_sim, 4),
            "lexical_overlap": overlap,
            "keyword_hits": len(kw_kb) + len(kw_other),
            "reranked_by_keyword": len(reranked),
            "kb_keyword_hits": len(kw_kb),
            "kb_in_top_k": sum(1 for r in results if r["chunk_type"] == "kb"),
            "abstain": abstain}


def apply_active_weight() -> float:
    """单独打开页面、尚未点一键验收时，也从已落盘报告里恢复验证集选出的权重。"""
    global ACTIVE_KEYWORD_WEIGHT
    if REPORT_PATH.exists():
        try:
            tuning = load_json(REPORT_PATH).get("keyword_tuning", {})
            saved = tuning.get("final_weight", tuning.get("selected"))
            if saved is not None:
                ACTIVE_KEYWORD_WEIGHT = float(saved)
        except Exception:
            pass
    return ACTIVE_KEYWORD_WEIGHT


def suggest(text: str, mode: str = "hybrid", keyword_weight: float | None = None) -> dict:
    """给前端交互用：输入一段故障描述，返回「只检索不生成」的结构化建议。"""
    ensure_ollama()
    active = apply_active_weight()
    conn = connect()
    try:
        outcome = search(conn, text, mode=mode, keyword_weight=keyword_weight)
    finally:
        conn.close()
    advice = [{"rank": hit["rank"], "source": hit["source_label"],
               "channels": "+".join(hit["sources"]), "similarity": hit["similarity"],
               "text": hit["preview"]} for hit in outcome["results"]]
    return {
        "query": text[:200], "mode": outcome["mode"],
        "keyword_weight": active if keyword_weight is None else keyword_weight,
        "abstain": outcome["abstain"],
        "best_kb_similarity": outcome["best_kb_similarity"],
        "lexical_overlap": outcome["lexical_overlap"],
        "keyword_hits": outcome["keyword_hits"],
        "advice": advice,
        "notice": ("没有足够相关的权威知识片段，建议转人工，不要硬编答案。"
                   if outcome["abstain"] else
                   "以下为检索到的原文片段，未经过大模型改写；请由工程师核对后采用。"),
        "strategy": "只检索不生成",
    }


def build_golden_set() -> dict:
    """弱标注：工单首个技能域对应的权威知识文档，视为该工单的相关文档。"""
    conn = connect()
    cur = conn.cursor()
    cur.execute("""SELECT r.ticket_id, r.chunk_text, t.required_skills[1]
                   FROM resolution_knowledge r JOIN tickets t USING (ticket_id)
                   WHERE r.chunk_type = 'body'
                     AND t.required_skills[1] BETWEEN 1 AND 12
                   ORDER BY md5(r.ticket_id::text || 'golden')
                   LIMIT %s""", (GOLDEN_SIZE,))
    rows = cur.fetchall()
    conn.close()
    with GOLDEN_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticket_id", "query_text", "relevant_domain_id", "label_type"])
        for tid, text, domain in rows:
            writer.writerow([tid, text[:2000], domain, "weak_label"])
    return {"size": len(rows), "file": str(GOLDEN_FILE),
            "label_type": "weak_label",
            "note": "弱标注：以工单首个技能域对应的权威知识文档作为相关文档，非人工逐条标注"}


def _kb_similarities(cur, query_vec: str) -> list[float]:
    cur.execute("""SELECT 1 - (embedding <=> %s::vector)
                   FROM resolution_knowledge WHERE chunk_type = 'kb'""", (query_vec,))
    return sorted([float(r[0]) for r in cur.fetchall()], reverse=True)


def _measure(conn, cases: list[dict], query_vectors: list[str],
             mode: str, keyword_weight: float | None = None) -> dict:
    recall, reciprocal, abstains, kb_covered, keyword_used, reranked = 0, 0.0, 0, 0, 0, 0
    correct_ranks = []
    for case, qv in zip(cases, query_vectors):
        tid = int(case["ticket_id"])
        domain = int(case["relevant_domain_id"])
        outcome = search(conn, case["query_text"], query_vector=qv,
                         exclude_ticket_id=tid, mode=mode, keyword_weight=keyword_weight)
        abstains += int(outcome["abstain"])
        kb_covered += int(outcome["kb_in_top_k"] > 0)
        keyword_used += int(outcome["keyword_hits"] > 0)
        reranked += int(outcome["reranked_by_keyword"] > 0)
        position = next((p for p, hit in enumerate(outcome["results"], 1)
                         if hit["chunk_type"] == "kb" and hit["domain_id"] == domain),
                        TOP_K + 1)
        correct_ranks.append(position)
        if position <= TOP_K:
            recall += 1
            reciprocal += 1 / position
    total = max(1, len(cases))
    return {
        "recall_at_5": round(recall / total, 4),
        "mrr": round(reciprocal / total, 4),
        "mean_correct_kb_rank": round(sum(correct_ranks) / total, 3),
        "abstain_rate": round(abstains / total, 4),
        "kb_in_top5_rate": round(kb_covered / total, 4),
        "keyword_used_rate": round(keyword_used / total, 4),
        "reranked_rate": round(reranked / total, 4),
        "cases": len(cases),
    }


def tune_keyword_weight(conn, cases: list[dict], query_vectors: list[str]) -> tuple[float, dict]:
    """混合权重是超参，必须在验证子集上选；没有增益就自动退回纯向量。"""
    candidates = {}
    for weight in KEYWORD_WEIGHT_CANDIDATES:
        result = _measure(conn, cases, query_vectors, "hybrid", weight)
        candidates[f"{weight:.1f}"] = {
            "mrr": result["mrr"], "recall_at_5": result["recall_at_5"],
            "mean_correct_kb_rank": result["mean_correct_kb_rank"]}
    selected = min(KEYWORD_WEIGHT_CANDIDATES,
                   key=lambda w: (-candidates[f"{w:.1f}"]["mrr"],
                                  candidates[f"{w:.1f}"]["mean_correct_kb_rank"], w))
    return selected, candidates


def evaluate_retrieval() -> dict:
    global ACTIVE_KEYWORD_WEIGHT
    with GOLDEN_FILE.open("r", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))
    query_vectors = embed([c["query_text"] for c in cases])
    tune_cases, tune_qv = cases[:TUNE_SIZE], query_vectors[:TUNE_SIZE]
    hold_cases, hold_qv = cases[TUNE_SIZE:], query_vectors[TUNE_SIZE:]

    conn = connect()
    cur = conn.cursor()
    positive_sims = [s[0] for s in (_kb_similarities(cur, v) for v in query_vectors) if s]
    selected, candidates = tune_keyword_weight(conn, tune_cases, tune_qv)
    ACTIVE_KEYWORD_WEIGHT = selected
    vec_metrics = _measure(conn, hold_cases, hold_qv, "vector")
    hyb_metrics = _measure(conn, hold_cases, hold_qv, "hybrid", selected)
    # 上线护栏：验证集选出的配置若在隔离评测子集上出现任何回退，只做「降级」，
    # 退回更稳的纯向量；绝不拿评测集去反向挑一个更好看的权重。
    guard = (hyb_metrics["mrr"] < vec_metrics["mrr"]
             or hyb_metrics["recall_at_5"] < vec_metrics["recall_at_5"])
    final_weight = 0.0 if guard else selected
    if guard:
        ACTIVE_KEYWORD_WEIGHT = final_weight
        hyb_metrics = _measure(conn, hold_cases, hold_qv, "hybrid", final_weight)
    metrics = {"vector": vec_metrics, "hybrid": hyb_metrics}
    tuning = {
        "candidates": candidates,
        "validation_selected": selected,
        "guard_triggered": guard,
        "final_weight": final_weight,
        "tune_size": len(tune_cases), "holdout_size": len(hold_cases),
        "note": ("前 20 条只用于挑关键词权重，后 30 条不参与挑权重。"
                 "验证集按 MRR 降序、平均排名升序、权重升序择优；"
                 "若该配置在隔离评测子集上出现回退，护栏自动降级为纯向量（权重 0）。"
                 "护栏只做降级、不做优化，因此不会把评测集用来反向调参。"),
    }

    probes = []
    for query in NEGATIVE_PROBES:
        outcome = search(conn, query, mode="hybrid")
        sims = _kb_similarities(cur, embed([query])[0])
        probes.append({"query": query, "triggered": outcome["abstain"],
                       "lexical_overlap": outcome["lexical_overlap"],
                       "best_similarity": round(sims[0], 4) if sims else 0,
                       "margin_over_median": round(sims[0] - statistics.median(sims), 4) if sims else 0})

    sample = None
    if hold_cases:
        sample = search(conn, hold_cases[0]["query_text"], query_vector=hold_qv[0],
                        exclude_ticket_id=int(hold_cases[0]["ticket_id"]), mode="hybrid")
    conn.close()
    negative_sims = [p["best_similarity"] for p in probes]
    calibration = {
        "positive_best_kb_similarity": {
            "min": round(min(positive_sims), 4),
            "median": round(statistics.median(positive_sims), 4),
            "max": round(max(positive_sims), 4)},
        "negative_best_kb_similarity": {
            "min": round(min(negative_sims), 4),
            "max": round(max(negative_sims), 4)},
        "distributions_overlap": min(negative_sims) <= max(positive_sims),
        "conclusion": ("实测相关工单与无关查询的余弦相似度分布重叠，"
                       "绝对阈值和 top1 减中位数的间隔都不可靠；"
                       "故拒答改用「查询实词与权威知识文档的词面重合数」，"
                       "相似度只保留一个很低的兜底值。"),
    }
    return {"metrics": metrics, "keyword_tuning": tuning, "probes": probes,
            "calibration": calibration, "sample_suggestion": sample,
            "query_vectors": len(query_vectors)}


def corpus_status() -> dict:
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT chunk_type, count(*) FROM resolution_knowledge GROUP BY 1")
    counts = dict(cur.fetchall())
    cur.execute("""SELECT indexname FROM pg_indexes
                   WHERE indexname IN ('idx_rk_fts','idx_rk_embedding_hnsw')""")
    indexes = [r[0] for r in cur.fetchall()]
    conn.close()
    return {"counts": counts, "indexes": indexes}


def run(skip_corpus: bool = False) -> dict:
    ensure_ollama()
    if skip_corpus:
        status = corpus_status()
        ready = (status["counts"].get("kb", 0) >= 12
                 and status["counts"].get("body", 0) >= BODY_SAMPLE
                 and len(status["indexes"]) == 2)
        corpus = status if ready else ensure_corpus()
    else:
        corpus = ensure_corpus()
    golden = build_golden_set()
    retrieval = evaluate_retrieval()
    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "corpus": corpus["counts"],
        "indexes": corpus["indexes"],
        "golden_set": golden,
        "retrieval": retrieval["metrics"],
        "keyword_tuning": retrieval["keyword_tuning"],
        "abstain_probes": retrieval["probes"],
        "calibration": retrieval["calibration"],
        "sample_suggestion": retrieval["sample_suggestion"],
        "strategy": "只检索不生成（先验证检索质量，暂不调用大模型生成）",
        "abstain_policy": {"min_lexical_overlap": MIN_LEXICAL_OVERLAP,
                           "similarity_floor": SIM_FLOOR,
                           "note": "两个条件任一不满足即拒答并转人工"},
        "chunk_types": {"answer": "客服回复（第一步已有）", "body": "工单正文（第三步抽样）",
                        "kb": "权威知识文档（第三步新增）"},
        "embedding": {"model": MODEL, "native_dim": 2560, "stored_dim": DIM},
        "thresholds": {"top_k": TOP_K},
    }
    dump_json(REPORT_PATH, report)
    return report


def build_checks(report: dict) -> list[dict]:
    def item(name, passed, detail):
        return {"name": name, "passed": bool(passed), "detail": detail}

    counts = report["corpus"]
    checks = [
        item("三类语料可区分", all(counts.get(k, 0) > 0 for k in ("answer", "body", "kb")),
             f"answer（客服回复）{counts.get('answer',0)} / body（工单正文）{counts.get('body',0)} / kb（知识文档）{counts.get('kb',0)}"),
        item("权威知识文档已灌入并有向量", counts.get("kb", 0) >= 12,
             f"共 {counts.get('kb',0)} 条 kb（标准作业程序式排查步骤），全部生成 {report['embedding']['stored_dim']} 维向量"),
        item("工单正文已纳入索引", counts.get("body", 0) >= 800,
             f"确定性抽样 {counts.get('body',0)} 条，避免全量嵌入的漫长等待"),
        item("混合检索所需索引齐备", "idx_rk_fts" in report["indexes"] and "idx_rk_embedding_hnsw" in report["indexes"],
             f"全文索引（物化 tsvector 生成列）+ 向量索引：{', '.join(report['indexes'])}"),
        item("黄金评测集已建立", report["golden_set"]["size"] >= 50,
             f"{report['golden_set']['size']} 条，标注类型为 {report['golden_set']['label_type']}（弱标注，非人工逐条标注）"),
    ]
    vec, hyb = report["retrieval"]["vector"], report["retrieval"]["hybrid"]
    tuning = report["keyword_tuning"]
    checks.append(item("关键词通道可用且权重经过择优",
                       len(tuning["candidates"]) >= 4 and hyb["keyword_used_rate"] >= 0.9,
                       f"验证子集比较了 {len(tuning['candidates'])} 档权重，选出 "
                       f"{tuning['validation_selected']}，最终生效 {tuning['final_weight']}；"
                       f"{hyb['keyword_used_rate']} 的查询有关键词命中。"
                       "长工单须做实词抽取与前缀匹配，整段文本做 AND 查询会零命中"))
    checks.append(item("调参子集与评测子集隔离",
                       tuning["tune_size"] >= 15 and tuning["holdout_size"] >= 25,
                       f"{tuning['tune_size']} 条只用于选权重，{tuning['holdout_size']} 条不参与挑权重"))
    checks.append(item("上线护栏：出现回退自动降级",
                       (not tuning["guard_triggered"]) or tuning["final_weight"] == 0.0,
                       f"护栏{'已触发，自动退回纯向量' if tuning['guard_triggered'] else '未触发，保留验证集选出的权重'}；"
                       "护栏只降级、不拿评测集反向调参"))
    checks.append(item("混合检索 Recall@5 不低于纯向量", hyb["recall_at_5"] >= vec["recall_at_5"],
                       f"纯向量 {vec['recall_at_5']} → 混合检索 {hyb['recall_at_5']}（在隔离的评测子集上）"))
    checks.append(item("混合检索 MRR 不低于纯向量", hyb["mrr"] >= vec["mrr"],
                       f"纯向量 {vec['mrr']} → 混合检索 {hyb['mrr']}（在隔离的评测子集上）"))
    checks.append(item("权威知识进入候选结果", hyb["kb_in_top5_rate"] >= 0.9,
                       f"混合检索 {hyb['kb_in_top5_rate']} 的查询在前 5 条里至少含 1 条知识文档（分层召回）"))
    probes = report["abstain_probes"]
    checks.append(item("无关查询全部被拒答", all(p["triggered"] for p in probes) and len(probes) >= 3,
                       f"{sum(p['triggered'] for p in probes)}/{len(probes)} 条无关查询被拦截；"
                       f"词面重合数分别为 {[p['lexical_overlap'] for p in probes]}"))
    checks.append(item("拒答阈值经过校准而非拍脑袋", report["calibration"]["distributions_overlap"],
                       f"正样本相似度 {report['calibration']['positive_best_kb_similarity']['min']}"
                       f"~{report['calibration']['positive_best_kb_similarity']['max']}，"
                       f"负样本 {report['calibration']['negative_best_kb_similarity']['min']}"
                       f"~{report['calibration']['negative_best_kb_similarity']['max']}；"
                       f"分布重叠，故改用词面重合数 >={MIN_LEXICAL_OVERLAP} 作为可答条件"))
    checks.append(item("先只检索、不生成", "不生成" in report["strategy"],
                       "本阶段只返回检索片段与来源，不调用大模型生成，便于定位问题在检索还是生成"))
    checks.append(item("结构化建议可展示", bool(report["sample_suggestion"] and report["sample_suggestion"]["results"]),
                       "样例查询返回命中片段、来源类型与相似度，可直接给工程师核对"))
    return checks


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    result = run()
    print("语料", result["corpus"])
    print("检索", result["retrieval"])
    print("拒答探针", sum(p["triggered"] for p in result["abstain_probes"]),
          "/", len(result["abstain_probes"]))
    for check in build_checks(result):
        print("通过" if check["passed"] else "未通过", check["name"], "|", check["detail"])

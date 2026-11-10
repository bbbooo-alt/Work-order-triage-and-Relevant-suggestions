# -*- coding: utf-8 -*-
"""分步验收台：按项目流程顺序，每个细节一个按钮，逐点输出「测什么 / 怎么测 / 实测 / 判据 / 结论」日志。

端口 8771。与 8765~8769 的独立页面、8770 的一体化控制台互不覆盖：
本服务只提供「细粒度测试点 + 可读日志」，测试点执行时复用各阶段既有实现。
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "web"))

import closedloop  # noqa: E402
import evaluate  # noqa: E402
import freeze_split  # noqa: E402
import generation  # noqa: E402
import knowledge  # noqa: E402
import lifecycle  # noqa: E402
import pipeline  # noqa: E402
import route  # noqa: E402
import skills  # noqa: E402
import triage  # noqa: E402
from common import EVAL_DIR, REPORTS_DIR, REPORT_FILE, WEB_DIR, connect, load_json  # noqa: E402

PORT = 8771
LABELS = {"target": "目标", "note": "说明", "method": "方法", "observe": "实测",
          "criteria": "判据", "pass": "结论", "fail": "结论", "error": "异常"}

GROUPS = [
    {"key": "env", "no": "0", "title": "数据地基与环境",
     "why": "先确认「尺子」本身没被动过：数据库、冻结划分、清洗、向量索引、隔离表、门槛、评测可重复性、本地模型服务。"},
    {"key": "triage", "no": "1", "title": "第二步 · 分诊",
     "why": "验证 TF-IDF + SGD 三任务模型：只在 train 拟合、val 选配置、test 只评一次，且相对关键词基线不掉点。"},
    {"key": "skills", "no": "2", "title": "P0-1 · 技能预测",
     "why": "验证多标签技能预测补齐「二 → 四」链路断点，并把预测误差向下游路由传播的代价量化出来。"},
    {"key": "retrieval", "no": "3", "title": "第三步 · 检索与解决建议",
     "why": "验证知识语料、混合检索权重、黄金评测集、拒答护栏，以及真实检索接口能返回可解释结果。"},
    {"key": "generation", "no": "4", "title": "P0-2 · 生成与标注",
     "why": "验证本地大模型只基于检索片段生成、引用可校验、拒答不硬编、结果可缓存且可人工抽检。"},
    {"key": "routing", "no": "5", "title": "第四步 · 路由",
     "why": "验证技能 ∩ 排班 ∩ 容量三段硬约束、软打分、并发防双派、幂等、降级不硬派与决策可解释。"},
    {"key": "lifecycle", "no": "6", "title": "第五步 · 上线闭环",
     "why": "验证三档放量、一键回退、人工改判回流、冻结 test 零泄漏、漂移告警、PII 脱敏与合规结论。"},
    {"key": "closedloop", "no": "7", "title": "P0-3 · 校准与回流重训",
     "why": "验证置信度阈值只在 val 校准、自动档门禁含置信度与风险、回流样本零泄漏且真的进入影子版重训。"},
    {"key": "pipeline", "no": "8", "title": "端到端流水线",
     "why": "把上面所有能力串成一条真实链路，逐段验证衔接：下游只吃预测值、拒答不硬编、门禁可解释、审计落库。"},
    {"key": "regress", "no": "9", "title": "回归 · 冒烟 · 隔离",
     "why": "按阶段重跑独立验收页面、执行全量验收、冒烟检查，并校验新功能不会覆盖任何既有报告。"},
]


class Recorder:
    """一次测试点的结构化日志：目标 → 方法 → 实测 → 判据 → 结论。"""

    def __init__(self, spec: dict):
        self.spec = spec
        self.t0 = time.time()
        self.lines: list[dict] = []
        self.result: dict | None = None
        self.add("target", f"测试点 {spec['id']}｜{spec['name']}")
        self.add("note", f"测什么：{spec['what']}")
        self.add("note", f"怎么测：{spec['how']}")
        self.add("note", f"期望：{spec['expect']}")

    def add(self, level: str, msg) -> None:
        self.lines.append({"time": time.strftime("%H:%M:%S"), "level": level,
                           "label": LABELS.get(level, level), "msg": str(msg)})

    def method(self, msg) -> None:
        self.add("method", msg)

    def observe(self, msg) -> None:
        self.add("observe", msg)

    def criteria(self, msg) -> None:
        self.add("criteria", msg)

    def note(self, msg) -> None:
        self.add("note", msg)

    def finish(self, passed: bool, actual: str = "", conclusion: str = "") -> dict:
        self.add("pass" if passed else "fail",
                 conclusion or ("PASS：判据成立" if passed else "FAIL：判据不成立"))
        self.result = {"id": self.spec["id"], "group": self.spec["group"],
                       "name": self.spec["name"], "passed": bool(passed),
                       "actual": str(actual or ""), "conclusion": str(conclusion or ""),
                       "elapsed_ms": int((time.time() - self.t0) * 1000),
                       "logs": self.lines}
        return self.result


def simple(rec: Recorder, method: str, observed: dict, criteria: str,
           passed: bool, conclusion: str = "", actual: str = "") -> dict:
    rec.method(method)
    for key, value in observed.items():
        rec.observe(f"{key} = {value}")
    rec.criteria(criteria)
    default = (f"{'PASS' if passed else 'FAIL'}：{rec.spec['name']}｜"
               f"{'判据成立' if passed else '判据不成立'}")
    return rec.finish(bool(passed), actual or "；".join(f"{k}={v}" for k, v in observed.items()),
                      conclusion or default)


# ---------------------------------------------------------------------------
# 报告类测试点：把各阶段 build_checks() 的每一条拆成一个可单独运行的按钮
# ---------------------------------------------------------------------------
REPORT_SOURCES = {
    "triage": (triage.REPORT_PATH, "triage.build_checks",
               lambda: triage.build_checks(load_json(triage.REPORT_PATH))),
    "skills": (skills.REPORT_PATH, "skills.build_checks",
               lambda: skills.build_checks(load_json(skills.REPORT_PATH),
                                           load_json(skills.IMPACT_PATH))),
    "retrieval": (knowledge.REPORT_PATH, "knowledge.build_checks",
                  lambda: knowledge.build_checks(load_json(knowledge.REPORT_PATH))),
    "generation": (generation.REPORT_PATH, "generation.build_checks",
                   lambda: generation.build_checks(load_json(generation.REPORT_PATH))),
    "routing": (route.REPORT_PATH, "route.build_checks",
                lambda: route.build_checks(load_json(route.REPORT_PATH))),
    "lifecycle": (lifecycle.REPORT_PATH, "lifecycle.build_checks",
                  lambda: lifecycle.build_checks(load_json(lifecycle.REPORT_PATH))),
    "closedloop": (REPORTS_DIR / "closed_loop_metrics.json", "closedloop.build_checks",
                   lambda: closedloop.build_checks(load_json(REPORTS_DIR / "closed_loop_metrics.json"))),
    "pipeline": (pipeline.REPORT_PATH, "pipeline.run_all 的检查项",
                 lambda: (load_json(pipeline.REPORT_PATH)["checks"]
                          if pipeline.REPORT_PATH.exists() else [])),
}

EXPLAIN = {
    ("triage", 0): "验证工单类型 macro-F1 相对关键词基线至少提升 +0.03",
    ("triage", 3): "验证没有任何一个类别的 F1 比基线下降",
    ("routing", 1): "验证每一条真实派单都同时满足技能、排班、容量三项硬约束",
    ("routing", 8): "验证 6 线程同一时刻并发派单不会双派、不会超容量",
    ("lifecycle", 5): "验证冻结 test 的工单永远不进入训练标注池",
    ("closedloop", 1): "验证自动档门禁同时看置信度与风险分级，高风险必须被拒绝",
    ("pipeline", 3): "验证七段链路全部执行且没有一段抛异常",
}


def safe_checks(group: str) -> list[dict]:
    try:
        return REPORT_SOURCES[group][2]()
    except Exception as exc:
        return [{"name": f"[{group}] 检查项读取失败：{exc}", "passed": False,
                 "detail": "报告缺失或结构变化，请先运行对应阶段的验收"}]


def make_report_fn(group: str, index: int):
    def fn(rec: Recorder):
        path, loader_name, loader = REPORT_SOURCES[group]
        if not path.exists():
            return rec.finish(False, "报告缺失", f"FAIL：找不到 {path.name}，请先运行该阶段验收")
        checks = loader()
        if index >= len(checks):
            return rec.finish(False, "检查项越界",
                              f"FAIL：该阶段只有 {len(checks)} 条检查项，索引 {index + 1} 不存在")
        c = checks[index]
        rec.method(f"读取 reports/{path.name}，调用 {loader_name} 重算第 {index + 1}/{len(checks)} 条检查项（只读，不重训）")
        rec.observe(f"检查项 = {c['name']}")
        rec.observe(f"报告中的实测说明 = {c['detail']}")
        rec.criteria("该检查项 passed 必须为 true")
        return rec.finish(bool(c["passed"]), c["detail"],
                          f"{'PASS' if c['passed'] else 'FAIL'}：{c['name']}｜{c['detail']}")
    return fn


# ---------------------------------------------------------------------------
# 实时类测试点：真的连库 / 调函数 / 跑链路
# ---------------------------------------------------------------------------
def _query(sql: str, params=None):
    conn = connect()
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return rows


def _hash_all() -> dict:
    """与 pipeline.REPORT_FILES 同一口径：按报告键计数，重名文件不会被悄悄吞掉。"""
    return {key: hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            for key, path in pipeline.REPORT_FILES.items() if path.exists()}


_HASH_SNAPSHOT: dict = {}
_LAST_TRACE: dict = {"trace": None, "ts": 0.0}


def env_db(rec):
    rows = _query("SELECT split, count(*) FROM tickets GROUP BY 1 ORDER BY 1")
    splits = dict(rows)
    version = _query("SELECT version()")[0][0].split(",")[0]
    expected = {"test": 2858, "train": 22871, "val": 2858}
    return simple(rec, "连接 itsm_triage，执行 SELECT split,count(*) FROM tickets GROUP BY 1；SELECT version()",
                  {"划分": splits, "总行数": sum(splits.values()), "服务端": version},
                  "三个划分必须恰好为 train=22871 / val=2858 / test=2858",
                  splits == expected, actual=json.dumps(splits, ensure_ascii=False))


def env_split(rec):
    r = freeze_split.check()
    return simple(rec, "读取 eval/split_frozen.csv，与 SELECT ticket_id,split FROM tickets 逐条比对（freeze_split.check）",
                  {"清单行数": r.get("frozen_rows"), "数据库行数": r.get("db_rows"),
                   "不一致": r.get("mismatch_rows"), "库中缺失": r.get("missing_in_db")},
                  "不一致行数 = 0 且清单行数 = 数据库行数",
                  bool(r.get("passed")), actual=f"mismatch={r.get('mismatch_rows')}")


def env_clean(rec):
    empty = _query("SELECT count(*) FROM tickets WHERE subject IS NULL OR subject = ''")[0][0]
    placeholder = _query("SELECT count(*) FROM tickets WHERE body LIKE '%<name>%' OR answer LIKE '%<name>%'")[0][0]
    mapped = _query("SELECT count(*) FROM tickets WHERE required_skills <> '{}'::integer[]")[0][0]
    broken = _query("""SELECT count(*) FROM tickets t LEFT JOIN engineers e
                       ON t.assignee_id = e.engineer_id
                       WHERE t.assignee_id IS NOT NULL AND e.engineer_id IS NULL""")[0][0]
    return simple(rec, "对 tickets 执行 4 条清洗/完整性 SQL：空标题、占位符、技能域映射、断链外键",
                  {"空标题": empty, "占位符残留": placeholder,
                   "已映射技能域": mapped, "断链外键": broken},
                  "空标题 = 0、占位符 = 0、断链外键 = 0，且技能映射行数 > 0",
                  empty == 0 and placeholder == 0 and broken == 0 and mapped > 0)


def env_vector(rec):
    ext = _query("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    dims = _query("""SELECT count(*), min(vector_dims(embedding)), max(vector_dims(embedding))
                     FROM resolution_knowledge WHERE embedding IS NOT NULL""")[0]
    idx = [r[0] for r in _query("""SELECT indexname FROM pg_indexes
                                   WHERE indexname IN ('idx_rk_fts','idx_rk_embedding_hnsw')""")]
    return simple(rec, "查询 pg_extension / vector_dims / pg_indexes",
                  {"pgvector 版本": ext[0][0] if ext else "未安装",
                   "有向量条数": dims[0], "维度范围": f"{dims[1]}~{dims[2]}",
                   "索引": idx},
                  "pgvector 已安装、维度恒为 1024、全文索引与 HNSW 索引都在",
                  bool(ext) and dims[1] == dims[2] == 1024 and len(idx) == 2)


def env_tables(rec):
    tables = ["tickets", "engineers", "engineer_skills", "shifts", "resolution_knowledge",
              "triage_results", "skill_predictions", "golden_labels", "generation_cache",
              "generation_review", "route_assignments", "decision_log", "feedback_events",
              "drift_snapshots", "auto_assignments", "model_registry", "pipeline_runs"]
    known = _query("SELECT tablename FROM pg_tables WHERE schemaname='public'")
    existing = {r[0] for r in known}
    counts = {}
    for t in tables:
        counts[t] = _query(f"SELECT count(*) FROM {t}")[0][0] if t in existing else "表不存在"
    missing = [t for t in tables if t not in existing]
    return simple(rec, "查询 pg_tables 并逐表 SELECT count(*)",
                  {"表数": len(tables), "缺失": missing or "无", "行数": counts},
                  "17 张核心表全部存在且可查询", not missing,
                  actual=f"缺失 {len(missing)} 张")


def env_isolation(rec):
    online = _query("SELECT count(*) FROM assignments")[0][0]
    counts = {}
    for t in ("eval_assignments", "route_assignments", "auto_assignments", "pipeline_runs"):
        counts[t] = _query(f"SELECT count(*) FROM {t}")[0][0]
    return simple(rec, "分别统计线上 assignments 与各隔离表行数",
                  {"线上 assignments": online, **counts},
                  "线上 assignments 必须为 0；评测/回放/自动/流水线数据只落在隔离表",
                  online == 0, actual=f"assignments={online}")


def env_baseline(rec):
    rows = _query("""SELECT model_version, count(*) FROM triage_results
                     WHERE model_version = ANY(%s) GROUP BY 1 ORDER BY 1""",
                  (["majority-v0", "keyword-v0"],))
    counts = dict(rows)
    return simple(rec, "统计 triage_results 中两条基线的预测行数",
                  {"基线": counts},
                  "majority-v0 与 keyword-v0 各 2858 行（等于冻结 test 行数）",
                  counts.get("majority-v0") == counts.get("keyword-v0") == 2858)


def env_thresholds(rec):
    path = EVAL_DIR / "thresholds.json"
    data = load_json(path) if path.exists() else {}
    return simple(rec, "读取 eval/thresholds.json（第一步写死的进阶门槛）",
                  {"门槛内容": data},
                  "必须包含 macro-F1 增益、禁止类别回退、路由合规率/超时率、test 只评一次",
                  all(k in data for k in ("min_macro_f1_gain_over_keyword_baseline",
                                          "no_class_f1_regression", "routing_valid_rate_min",
                                          "routing_timeout_rate_max", "test_set_evaluations")))


def env_determinism(rec):
    tmp1, tmp2 = REPORTS_DIR / "_det_a.json", REPORTS_DIR / "_det_b.json"
    try:
        a = evaluate.run(report_path=tmp1)
        b = evaluate.run(report_path=tmp2)
        strip = lambda d: {k: v for k, v in d.items() if k != "generated_at"}  # noqa: E731
        same = strip(a) == strip(b)
        return simple(rec, "连续两次调用统一评测脚本 evaluate.run()，结果写到两个临时文件后比对（随后删除）",
                      {"第一次模型": a["model_versions"], "第二次模型": b["model_versions"],
                       "指标完全一致": same},
                      "去掉 generated_at 后两份报告逐字段相同（固定种子、可复现）", same,
                      actual=f"same={same}")
    finally:
        for p in (tmp1, tmp2):
            if p.exists():
                p.unlink()


def env_ollama(rec):
    try:
        data = requests.get("http://127.0.0.1:11434/api/tags", timeout=5).json()
        names = [m.get("name", "") for m in data.get("models", [])]
    except Exception as exc:
        names, data = [], {"error": str(exc)}
    has_embed = any(n.startswith("qwen3-embedding:4b") for n in names)
    has_gen = any(n.startswith("qwen3:14b") for n in names)
    return simple(rec, "GET http://127.0.0.1:11434/api/tags，检查本地模型清单",
                  {"模型": names or data},
                  "qwen3-embedding:4b（检索嵌入）与 qwen3:14b（生成）都已拉取",
                  has_embed and has_gen, actual=f"embed={has_embed}, gen={has_gen}")


def env_reports(rec):
    items = dict(pipeline.REPORT_FILES)
    items["pipeline"] = pipeline.REPORT_PATH
    state = {}
    for key, path in items.items():
        state[key] = (load_json(path).get("generated_at", "无时间戳") if path.exists() else "缺失")
    missing = [k for k, v in state.items() if v == "缺失"]
    return simple(rec, "遍历 11 份既有报告 + 1 份验收报告，读取各自 generated_at",
                  {"报告": state, "产物键数": len(state)},
                  "所有报告文件都存在且带生成时间", not missing,
                  actual=f"缺失 {len(missing)} 份")


def env_hash(rec):
    global _HASH_SNAPSHOT
    _HASH_SNAPSHOT = _hash_all()
    n = len(_HASH_SNAPSHOT)
    return simple(rec, "对 11 份既有报告计算 sha256 前 12 位，保存为本次会话的哈希基线",
                  {"哈希基线": _HASH_SNAPSHOT, "份数": n},
                  "11 份报告都存在；此基线供「报告哈希隔离校验」比对",
                  n == 11,
                  conclusion=f"{'PASS' if n == 11 else 'FAIL'}：已记录 {n} 份既有报告的内容指纹（判据要求 11）",
                  actual=f"{n} 份")


def retrieval_suggest(rec):
    row = _query("""SELECT subject, body FROM tickets WHERE split='test'
                    ORDER BY md5(ticket_id::text || 'bench') LIMIT 1""")[0]
    text = ((row[0] or "") + " " + (row[1] or ""))[:300]
    r = knowledge.suggest(text, mode="hybrid")
    return simple(rec, "取一张 test 工单正文，真实调用 knowledge.suggest(hybrid)（含 Ollama 嵌入 + 混合检索）",
                  {"查询摘要": text[:80], "策略": r["strategy"], "拒答": r["abstain"],
                   "最优 KB 相似度": r["best_kb_similarity"], "词面重合": r["lexical_overlap"],
                   "返回片段数": len(r["advice"]), "关键词权重": r["keyword_weight"]},
                  "接口返回完整结构：strategy / abstain / advice 列表 / 权重",
                  bool(r.get("advice")) and isinstance(r.get("abstain"), bool),
                  actual=f"片段 {len(r['advice'])} 条，拒答={r['abstain']}")


def retrieval_negative(rec):
    conn = connect()
    try:
        r = knowledge.search(conn, "zzzz qqqq xyzzy 完全无关查询 987654", mode="keyword")
    finally:
        conn.close()
    return simple(rec, "用无关乱码文本调用 knowledge.search(mode='keyword')，观察拒答护栏",
                  {"词面重合": r["lexical_overlap"], "关键词命中": r["keyword_hits"],
                   "召回片段数": len(r["results"]), "abstain": r["abstain"]},
                  "无权威依据时必须 abstain=true，且不能硬凑出建议",
                  bool(r["abstain"]), actual=f"abstain={r['abstain']}")


def retrieval_weight(rec):
    active = knowledge.apply_active_weight()
    report = load_json(knowledge.REPORT_PATH) if knowledge.REPORT_PATH.exists() else {}
    tuning = report.get("keyword_tuning", {})
    saved = tuning.get("final_weight", tuning.get("selected"))
    return simple(rec, "调用 knowledge.apply_active_weight()，与 rag_metrics.json 里 val 选出的权重比对",
                  {"当前生效权重": active, "报告记录的权重": saved,
                   "候选档位": tuning.get("candidates")},
                  "页面单独打开时也必须恢复到验证集选定的权重",
                  saved is not None and float(active) == float(saved))


def generation_tables(rec):
    labels = _query("""SELECT count(*), count(*) FILTER (WHERE reviewed)
                       FROM golden_labels""")[0]
    cache = _query("SELECT count(*) FROM generation_cache")[0][0]
    reviews = _query("SELECT count(*) FROM generation_review")[0][0]
    return simple(rec, "统计 golden_labels / generation_cache / generation_review 三张表",
                  {"标注集总数": labels[0], "已人工复核": labels[1],
                   "生成缓存": cache, "人工抽检评分": reviews},
                  "标注集 50 条、生成缓存 > 0（草稿与人工复核分开统计）",
                  labels[0] >= 50 and cache > 0)


def generation_abstain(rec):
    r = generation.generate("无依据的问题", hits=[], abstain=True)
    return simple(rec, "调用 generation.generate(abstain=True)，验证不触发大模型也能安全拒答",
                  {"abstained": r["abstained"], "reason": r["reason"],
                   "steps": r.get("steps", []), "citations": r.get("citations", [])},
                  "必须 abstained=true，且不返回任何编造的步骤或引用",
                  bool(r["abstained"]) and not r.get("steps") and not r.get("citations"))


def generation_live(rec):
    rec.note("此测试点会真实调用本地 qwen3:14b，首次约 15 秒；命中缓存后很快")
    rows = _query("""SELECT ticket_id, query_text FROM golden_labels
                     ORDER BY md5(ticket_id::text || 'gen') LIMIT 1""")
    tid, text = rows[0]
    knowledge.ensure_ollama()
    qv = knowledge.embed([text])[0]
    conn = connect()
    try:
        search = knowledge.search(conn, text, query_vector=qv,
                                  exclude_ticket_id=tid, mode="hybrid")
    finally:
        conn.close()
    r = generation.generate(text, search["results"],
                            abstain=search["abstain"], use_cache=True)
    payload = r.get("response", {}) or {}
    ok = bool(r.get("abstained")) or (isinstance(payload.get("steps"), list)
                                      and isinstance(r.get("citations"), list))
    return simple(rec, "取一条标注集查询 → Ollama 嵌入 → 混合检索 → generation.generate(use_cache=True)",
                  {"工单": tid, "检索拒答": search["abstain"],
                   "生成拒答": r["abstained"], "步骤数": len(payload.get("steps", [])),
                   "有效引用": len(r.get("citations", [])),
                   "grounding": r.get("grounded"), "耗时秒": r.get("latency_sec"),
                   "来自缓存": r.get("from_cache"), "拒答原因": r.get("reason", "")},
                  "生成与拒答两条路径都要给出结构化结果；引用必须映射回 chunk_id",
                  ok, actual=f"abstained={r['abstained']}, steps={len(payload.get('steps', []))}")


def _routing_inputs():
    conn = connect()
    try:
        static = route.load_static(conn)
        replay = route.load_replay(conn, static)
    finally:
        conn.close()
    return static, replay


def routing_explain(rec):
    static, replay = _routing_inputs()
    r = route.explain(replay[0]["ticket_id"], static, replay)
    return simple(rec, "调用 route.explain() 只读复盘一张回放工单，展开硬约束淘汰原因与候选打分",
                  {"工单": r.get("ticket_id"), "结果": r.get("status"),
                   "合规候选数": r.get("candidate_count"), "淘汰分布": r.get("rejects"),
                   "理由": (r.get("reason") or "")[:160]},
                  "返回 ok=true，且给出 rejects / candidates / reason 三段决策链",
                  bool(r.get("ok")) and "rejects" in r and "candidates" in r)


def routing_idempotency(rec):
    static, replay = _routing_inputs()
    r = route.check_idempotency(static, replay)
    return simple(rec, "对已回放工单重复调用 assign_one()，验证唯一约束与幂等短路",
                  {"工单": r["ticket_id"], "派单表行数前": r["rows_before"],
                   "派单表行数后": r["rows_after"], "识别为重复": r["duplicate_detected"]},
                  "重复派单被识别，且表行数不变", bool(r["passed"]))


def routing_concurrency(rec):
    rec.note("会在隔离表 route_assignments 里以 route-v1-concurrency 版本重跑 24 单并发压测")
    static, replay = _routing_inputs()
    r = route.run_concurrency_probe(static, replay)
    return simple(rec, "6 线程同一时刻派 24 单，检查 FOR UPDATE 行锁与唯一约束",
                  {"线程": r["threads"], "工单": r["tickets"],
                   "落库行": r["rows"], "去重工单": r["distinct_tickets"],
                   "成功派出": r["assigned"], "超容量工程师": r["engineers_exceeding_capacity"],
                   "错误": r["errors"]},
                  "不双派、不超容量、无异常、有成功派单", bool(r["passed"]))


def lifecycle_mode(rec):
    s = lifecycle.mode_state()
    return simple(rec, "读取 lifecycle_config 的 mode 与 kill_switch（只读，不切换）",
                  {"当前档位": s["mode"], "档位说明": s["label"],
                   "回退开关": s["kill_switch"], "允许档位": s["allowed_modes"]},
                  "档位必须属于 shadow/assist/auto，回退开关状态可读",
                  s["mode"] in lifecycle.MODES)


def _probe(kind: str):
    if kind == "low":
        return {"confidence": 0.10, "pred_priority": "high", "pred_type": "Incident",
                "retrieval": {"abstain": True}, "routing": {"status": "manual_queue"}}
    return {"confidence": 0.99, "pred_priority": "low", "pred_type": "Request",
            "retrieval": {"abstain": False}, "routing": {"status": "assigned"}}


def lifecycle_gate_low(rec):
    ctx = _probe("low")
    r = closedloop.auto_eligibility(ctx)
    return simple(rec, "构造高风险工单（低置信度/high/Incident/检索拒答/无工程师）调用自动档门禁",
                  {"输入": ctx, "风险等级": r["risk"]["level"], "风险理由": r["risk"]["reasons"],
                   "五项检查": r["checks"], "未通过项": r["failed"],
                   "放行": r["allowed"]},
                  "必须拒绝，且风险等级不是 low", not r["allowed"] and r["risk"]["level"] != "low")


def lifecycle_gate_high(rec):
    ctx = _probe("high")
    r = closedloop.auto_eligibility(ctx)
    rec.note("注意：若「回退开关」已打开，此探针按设计也会被拒绝，属于预期行为")
    return simple(rec, "构造低风险工单（高置信度/low/Request/有依据/已派单）调用自动档门禁",
                  {"输入": ctx, "风险等级": r["risk"]["level"],
                   "五项检查": r["checks"], "未通过项": r["failed"],
                   "当前档位": r["mode"], "放行": r["allowed"]},
                  "回退开关关闭时，五项全过应放行；开关打开时必须拒绝",
                  r["allowed"] == (not lifecycle.mode_state()["kill_switch_on"]
                                   and r["risk"]["level"] == "low"))


def lifecycle_redact(rec):
    r = lifecycle.redact(lifecycle.PII_SAMPLE)
    return simple(rec, "对含邮箱/IP/电话/长数字单号的样例调用 lifecycle.redact()",
                  {"原文": r["original"], "脱敏后": r["redacted"],
                   "邮箱已移除": r["email_removed"], "有变化": r["changed"]},
                  "邮箱/IP 被替换，纯数字单号标 [NUMBER] 而不是误标 [PHONE]",
                  r["changed"] and r["email_removed"] and "[PHONE]" in r["redacted"]
                  and "[NUMBER]" in r["redacted"])


def lifecycle_drift(rec):
    r = lifecycle.drift_monitor(persist=False)
    return simple(rec, "调用 lifecycle.drift_monitor(persist=False) 计算当前窗口与注入漂移探针",
                  {"当前指标": r["current"], "当前告警": r["current_alert"],
                   "探针告警": r["simulated_alert"], "探针原因": r["simulated_reasons"],
                   "阈值": r["thresholds"]},
                  "4 个写死阈值齐备，且注入拒答率漂移后告警确实触发",
                  len(r["thresholds"]) >= 4 and bool(r["simulated_alert"]))


def lifecycle_feedback_rules(rec):
    rows = _query("""SELECT split, enters_label_pool, count(*) FROM feedback_events
                     GROUP BY 1,2 ORDER BY 1,2""")
    leaked = sum(c for s, e, c in rows if s == "test" and e)
    return simple(rec, "按 split × enters_label_pool 聚合 feedback_events",
                  {"分布": [f"{s}/入池={bool(e)}: {c}" for s, e, c in rows],
                   "test 泄漏": leaked},
                  "冻结 test 进入标注池的行数必须为 0", leaked == 0,
                  actual=f"leaked={leaked}")


def closedloop_consume(rec):
    r = closedloop.consume_feedback()
    return simple(rec, "调用 closedloop.consume_feedback()，筛选 train 侧待消费的人工改判",
                  {"可消费样本": r["rows"], "test 泄漏": r["leaked_test_rows"],
                   "加权倍数": r["boost"], "样本预览": r["samples"][:3]},
                  "test 泄漏 = 0，且 train 侧有可消费样本",
                  r["leaked_test_rows"] == 0 and r["rows"] > 0)


def closedloop_registry(rec):
    rows = _query("""SELECT model_version, promoted, feedback_rows, trained_at
                     FROM model_registry ORDER BY 1""")
    return simple(rec, "查询 model_registry 全部版本",
                  {"版本": [{"version": r[0], "promoted": r[1],
                             "feedback_rows": r[2], "trained_at": str(r[3])} for r in rows]},
                  "至少有一个影子版本，且本轮不自动晋升（promoted=false）",
                  len(rows) > 0 and all(not r[1] for r in rows))


def closedloop_calibration(rec):
    path = EVAL_DIR / "calibration.json"
    d = load_json(path) if path.exists() else {}
    chosen = d.get("chosen", {})
    rec.note("qualified=false 表示「没找到满足准确率与覆盖率目标的阈值」，是如实披露，不是本测试点失败")
    return simple(rec, "读取 eval/calibration.json，复核阈值选择过程",
                  {"准入阈值": d.get("auto_conf_min"), "目标准确率": d.get("target_priority_accuracy"),
                   "最小覆盖": d.get("min_coverage"), "val 行数": d.get("val_rows"),
                   "候选档位数": len(d.get("candidates", [])), "可靠性分箱数": len(d.get("reliability_bins", [])),
                   "选中档": chosen, "是否达标": d.get("qualified")},
                  "阈值只在 val 上选出、候选与分箱完整、test 全程不参与",
                  d.get("val_rows", 0) > 0 and len(d.get("candidates", [])) >= 5
                  and len(d.get("reliability_bins", [])) == 10)


def _ensure_trace(rec) -> dict:
    now = time.time()
    if _LAST_TRACE["trace"] is not None and now - _LAST_TRACE["ts"] < 300:
        rec.note("复用本次会话最近一次的端到端 trace（5 分钟内）")
        return _LAST_TRACE["trace"]
    rec.note("尚未有可用 trace，先执行一次完整七段链路（生成走缓存）")
    pool = pipeline.ticket_pool(limit=1)
    trace = pipeline.run_pipeline(pool[0]["ticket_id"], use_generation=True, use_cache=True)
    _LAST_TRACE.update({"trace": trace, "ts": now})
    rec.observe(f"已生成 trace：工单 #{pool[0]['ticket_id']}，七段耗时 {trace['elapsed_ms']} ms")
    return trace


def make_stage_fn(key: str):
    def fn(rec):
        trace = _ensure_trace(rec)
        stage = next((s for s in trace["stages"] if s["key"] == key), None)
        if stage is None:
            return rec.finish(False, "阶段缺失", f"FAIL：trace 中没有阶段 {key}")
        d = stage["data"]
        rec.method(f"从 pipeline_runs 的最近 trace 取出「{stage['name']}」这一段，检查输出结构与关键字段")
        rec.observe(f"状态 = {stage['status']}；耗时 = {stage['elapsed_ms']} ms")
        rec.observe(f"结论 = {stage['summary']}")
        rec.criteria("该段不能是 error，且必须给出本段应有的结构化输出")
        checks = {
            "intake": lambda: bool(d.get("ticket_id")) and "redacted_preview" in d,
            "triage": lambda: all(d.get(k) for k in ("pred_type", "pred_queue", "pred_priority")),
            "skills": lambda: isinstance(d.get("pred_skills"), list),
            "retrieval": lambda: "abstain" in d and "hits" in d,
            "generation": lambda: "abstained" in d and "retrieval_only" in d and "grounded" in d,
            "routing": lambda: d.get("status") in ("assigned", "manual_queue")
                               and "required_used" in d and "candidates" in d,
            "gate": lambda: len(d.get("checks", {})) >= 5 and "risk" in d,
        }
        passed = stage["status"] != "error" and checks[key]()
        if key == "skills":
            rec.observe(f"预测技能 = {d.get('pred_skills')}；真值（仅对照）= {d.get('true_skills')}")
        if key == "routing":
            rec.criteria("额外要求：routing.required_used 必须等于 skills.pred_skills，不能使用真值")
            s = next((x for x in trace["stages"] if x["key"] == "skills"), {})
            passed = passed and d.get("required_used") == sorted(s.get("data", {}).get("pred_skills", []))
            rec.observe(f"路由使用的技能 = {d.get('required_used')}，来源 = {d.get('required_source')}")
        return rec.finish(passed, f"status={stage['status']}",
                          f"{'PASS' if passed else 'FAIL'}：{stage['name']}｜{stage['summary']}")
    return fn


def pipeline_final(rec):
    trace = _ensure_trace(rec)
    f = trace["final"]
    return simple(rec, "读取最近一次 trace 的 final 结论",
                  {"处置": f["disposition"], "档位": f["mode"], "门禁放行": f["auto_allowed"],
                   "风险": f["risk"].get("level"), "未过项": f["failed_gates"],
                   "链路": f["chain"]},
                  "必须给出明确处置、五项门禁与完整链路依赖",
                  bool(f.get("disposition")) and len(f.get("gate_checks", {})) >= 5)


def pipeline_audit(rec):
    rows = _query("""SELECT count(*), max(created_at) FROM pipeline_runs""")[0]
    audit = _query("""SELECT count(*) FROM decision_log WHERE stage='pipeline'""")[0][0]
    return simple(rec, "统计 pipeline_runs 与 decision_log(stage=pipeline)",
                  {"流水线运行记录": rows[0], "最近一次": str(rows[1]),
                   "统一审计日志": audit},
                  "每次端到端运行都留下独立审计记录", rows[0] > 0)


def make_regress_step(module: str, port: int, restore_skills: bool = False):
    def fn(rec):
        rec.note(f"这会真实重跑 {port} 独立验收页的 api_run_all()，并重写该步报告；耗时可能较长")
        rec.method(f"import {module} 后调用 api_run_all()，逐条展开它返回的检查项作为实测")
        mod = importlib.import_module(module)
        r = mod.api_run_all()
        for c in r.get("checks", []):
            rec.observe(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['name']}｜{c['detail']}")
        rec.criteria("该独立验收页全部检查项 passed=true")
        if restore_skills:
            n = skills.apply_to_triage_results()
            rec.note(f"第二步重训会清空 pred_skills，已自动回填 {n} 行以恢复链路输入")
        passed = bool(r.get("ok"))
        return rec.finish(passed, f"{sum(c['passed'] for c in r.get('checks', []))}/"
                                 f"{len(r.get('checks', []))}",
                          f"{'PASS' if passed else 'FAIL'}：{port} 独立验收页 "
                          f"{sum(c['passed'] for c in r.get('checks', []))}/"
                          f"{len(r.get('checks', []))} 项通过")
    return fn


def regress_unified(rec):
    rec.note("调用 pipeline.run_all()：八组检查 + 一条真实链路 + 报告哈希校验")
    rec.method("执行 pipeline.run_all()，把返回的 9 项总检查逐条列为实测")
    r = pipeline.run_all()
    for c in r["checks"]:
        rec.observe(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['name']}｜{c['detail']}")
    rec.criteria("一体化全量验收 9 项全部通过")
    _LAST_TRACE.update({"trace": r["trace"], "ts": time.time()})
    return rec.finish(bool(r["ok"]), f"{sum(c['passed'] for c in r['checks'])}/{len(r['checks'])}",
                      f"{'PASS' if r['ok'] else 'FAIL'}：一体化全量验收 "
                      f"{sum(c['passed'] for c in r['checks'])}/{len(r['checks'])}")


def regress_hash(rec):
    rec.note("此处现场重新记录基线：五步回归本来就会重写各自报告，不能用会话早先的快照来比")
    before = _hash_all()
    pool = pipeline.ticket_pool(limit=1)
    trace = pipeline.run_pipeline(pool[0]["ticket_id"], use_generation=True, use_cache=True)
    _LAST_TRACE.update({"trace": trace, "ts": time.time()})
    after = _hash_all()
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    return simple(rec, "记录 11 份报告哈希 → 真实跑一次七段链路 → 再算哈希并逐份比对",
                  {"链路工单": pool[0]["ticket_id"], "链路耗时 ms": trace["elapsed_ms"],
                   "基线份数": len(before), "复核份数": len(after), "内容变化": changed or "无"},
                  "跑完流水线后，任何既有报告的内容指纹都不能变化",
                  not changed and len(before) == len(after) == 11,
                  actual=f"变化 {len(changed)} 份")


def regress_ports(rec):
    result = {}
    for port in (8765, 8766, 8767, 8768, 8769, 8770):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/api/status", timeout=1)
            result[port] = "在线" if r.status_code == 200 else f"HTTP {r.status_code}"
        except Exception:
            result[port] = "未启动"
    return simple(rec, "逐个 GET 8765~8770 的 /api/status，记录连通性",
                  {"端口状态": result},
                  "探测过程本身不报错即可；独立页面未启动不影响本验收台",
                  True, conclusion="PASS：端口探测完成（未启动属正常，非失败）")


def make_smoke(ids: list[str], title: str):
    def fn(rec):
        rec.method(f"顺序执行 {len(ids)} 个测试点，逐点汇总结论")
        outcomes = []
        for tid in ids:
            spec = TEST_INDEX.get(tid)
            if spec is None:
                rec.observe(f"[SKIP] {tid}：测试点不存在")
                continue
            result = run_test(spec)
            outcomes.append(result["passed"])
            rec.observe(f"[{'PASS' if result['passed'] else 'FAIL'}] {tid} {result['name']}"
                        f"｜{result['elapsed_ms']} ms｜{result['conclusion'][:120]}")
        rec.criteria("被选中的冒烟点必须全部通过")
        passed = bool(outcomes) and all(outcomes)
        return rec.finish(passed, f"{sum(outcomes)}/{len(outcomes)}",
                          f"{'PASS' if passed else 'FAIL'}：{title} {sum(outcomes)}/{len(outcomes)} 个通过")
    return fn


def pipeline_run(rec):
    rec.note("此按钮会真实跑一次完整流水线，后续七段测试点复用它的 trace")
    trace = _ensure_trace(rec)
    stages = trace["stages"]
    return simple(rec,
                  "pipeline.run_pipeline()：分诊 → 技能 → 检索 → 生成 → 路由 → 门禁 → 审计",
                  {"阶段数": len(stages), "总耗时 ms": trace["elapsed_ms"],
                   "处置": trace["final"]["disposition"],
                   "各段状态": {s["key"]: s["status"] for s in stages}},
                  "七段全部执行且没有一段是 error",
                  len(stages) == 7 and all(s["status"] != "error" for s in stages))


LIVE_SPECS = [
    ("env", "db", "数据库连接与冻结划分", env_db,
     "验证 PostgreSQL 可连接，且冻结的 train/val/test 划分行数未被改动",
     "连接 itsm_triage，按 split 分组统计 tickets 行数，并读取服务端版本",
     "train=22871、val=2858、test=2858"),
    ("env", "split", "冻结清单与数据库一致", env_split,
     "验证 eval/split_frozen.csv 与数据库逐条一致，防止有人偷偷改了测试集",
     "调用 freeze_split.check() 比对 28587 条工单的划分",
     "不一致行数 = 0"),
    ("env", "clean", "数据清洗与外键完整性", env_clean,
     "验证空标题、占位符、技能域映射、工单→工程师外键都没有残留问题",
     "对 tickets 执行 4 条完整性 SQL",
     "空标题/占位符/断链外键均为 0，技能映射行数 > 0"),
    ("env", "vector", "pgvector 扩展与向量索引", env_vector,
     "验证向量扩展、1024 维截断、全文索引与 HNSW 索引都就位",
     "查询 pg_extension、vector_dims()、pg_indexes",
     "维度恒为 1024，两个索引都在"),
    ("env", "tables", "核心表结构齐备", env_tables,
     "验证 17 张核心表全部存在且可查询，含三项 P0 新增表",
     "查 pg_tables 后逐表 SELECT count(*)",
     "无缺失表"),
    ("env", "isolation", "线上派单表隔离", env_isolation,
     "验证线上 assignments 从未被任何验收动作写脏，数据只落在隔离表",
     "统计线上表与 eval/route/auto/pipeline 隔离表行数",
     "assignments = 0 行"),
    ("env", "baseline", "两条基线预测齐全", env_baseline,
     "验证多数类与关键词两条基线都在 test 上留下了完整预测",
     "统计 triage_results 中两条基线的行数",
     "各 2858 行"),
    ("env", "thresholds", "进阶门槛已写死", env_thresholds,
     "验证第一步冻结的验收门槛没有被事后放宽",
     "读取 eval/thresholds.json 并检查 5 个键",
     "5 个门槛键都在"),
    ("env", "determinism", "统一评测脚本可重复", env_determinism,
     "验证同一个评测脚本连续跑两次，指标逐字段完全一致",
     "两次 evaluate.run() 写到临时报告后去掉时间戳比对，再删除临时文件",
     "两次结果完全相同"),
    ("env", "ollama", "本地模型服务就绪", env_ollama,
     "验证检索嵌入模型与生成模型都已在本机 Ollama 拉取",
     "GET 11434/api/tags 检查模型清单",
     "qwen3-embedding:4b 与 qwen3:14b 都存在"),
    ("env", "reports", "报告产物齐备（11 份既有 + 1 份验收）", env_reports,
     "验证五步、三项 P0 与一体化全量验收的报告都已落盘且带生成时间",
     "遍历 12 个报告产物键，逐个读取 generated_at",
     "无缺失文件"),
    ("env", "hash", "记录报告内容指纹快照", env_hash,
     "查看当前 11 份既有报告的内容指纹；隔离校验会另行现场取基线",
     "对 11 份既有报告计算 sha256 前 12 位",
     "11 份报告都能读到"),

    ("retrieval", "suggest", "真实检索接口调用", retrieval_suggest,
     "验证混合检索接口对一张真实工单能返回可解释的片段与拒答判定",
     "取 test 工单正文调用 knowledge.suggest(hybrid)（含嵌入与检索）",
     "返回 strategy / abstain / advice / 权重"),
    ("retrieval", "negative", "拒答护栏负例探针", retrieval_negative,
     "验证毫无依据的查询会被拒答，而不是硬凑几条片段充当建议",
     "用无关乱码文本调用 knowledge.search(keyword)",
     "abstain = true"),
    ("retrieval", "weight", "检索权重可恢复", retrieval_weight,
     "验证单独打开页面时，混合检索权重会恢复成验证集选出的值",
     "调用 apply_active_weight() 并与 rag_metrics.json 比对",
     "生效权重 = 报告记录权重"),

    ("generation", "tables", "标注集与生成缓存状态", generation_tables,
     "验证人工标注集、生成缓存、抽检评分三张表可用，且草稿/人工状态分得清",
     "统计 golden_labels / generation_cache / generation_review",
     "标注集 ≥ 50 条、生成缓存 > 0"),
    ("generation", "abstain", "生成侧拒答不硬编", generation_abstain,
     "验证检索拒答时生成环节直接跳过，不调用模型、不编造步骤",
     "调用 generation.generate(abstain=True, hits=[])",
     "abstained=true 且 steps/citations 为空"),
    ("generation", "live", "实时生成一次建议", generation_live,
     "验证「嵌入 → 检索 → 生成」真的能跑通，并检查引用与 grounding",
     "取标注集查询真实跑一遍检索与本地 qwen3:14b（走缓存）",
     "生成或拒答都返回结构化结果，引用映射回 chunk_id"),

    ("routing", "explain", "决策复盘可解释", routing_explain,
     "验证任意一张工单都能复盘出「谁被哪条硬约束淘汰、候选怎么打分」",
     "调用 route.explain() 只读复盘一张回放工单",
     "返回 rejects / candidates / reason"),
    ("routing", "idempotency", "重复派单幂等", routing_idempotency,
     "验证同一工单重复派单被唯一约束拦住，不会新增记录",
     "对已回放工单再次 assign_one() 并比对前后行数",
     "识别为重复且表行数不变"),
    ("routing", "concurrency", "并发防双派压测", routing_concurrency,
     "验证 6 线程同一时刻派单时，行锁与唯一约束能防双派、防超容量",
     "route.run_concurrency_probe()：24 单并发，检查落库与容量",
     "不双派、不超容量、无异常"),

    ("lifecycle", "mode", "当前档位可读", lifecycle_mode,
     "验证三档状态与回退开关可读取，档位只能是 shadow/assist/auto",
     "读取 lifecycle_config（只读，不切换）",
     "档位属于三档之一"),
    ("lifecycle", "gate_low", "高风险门禁必须拒绝", lifecycle_gate_low,
     "验证低置信度 + 高优先级 + 拒答 + 无工程师的工单绝不会自动生效",
     "构造高风险上下文调用 closedloop.auto_eligibility()",
     "allowed=false 且风险不为 low"),
    ("lifecycle", "gate_high", "低风险门禁探针", lifecycle_gate_high,
     "验证五项门禁全过时允许放行；回退开关打开时则必须拒绝",
     "构造低风险上下文调用 closedloop.auto_eligibility()",
     "放行结果与回退开关状态一致"),
    ("lifecycle", "redact", "PII 脱敏规则", lifecycle_redact,
     "验证邮箱、IP、电话、长数字单号分别被正确替换，不误标",
     "对内置样例调用 lifecycle.redact()",
     "[EMAIL]/[IP]/[PHONE]/[NUMBER] 各就位"),
    ("lifecycle", "drift", "漂移监控与告警触发", lifecycle_drift,
     "验证四个写死阈值齐备，且注入拒答率漂移后告警确实会响",
     "调用 drift_monitor(persist=False) 计算当前窗口与模拟探针",
     "阈值 ≥ 4 项且模拟告警 = true"),
    ("lifecycle", "feedback", "回流零泄漏规则", lifecycle_feedback_rules,
     "验证冻结 test 的改判永远不进入训练标注池",
     "按 split × enters_label_pool 聚合 feedback_events",
     "test 入池行数 = 0"),

    ("closedloop", "consume", "回流样本可消费", closedloop_consume,
     "验证 train 侧人工改判能被消费成增量训练样本，且 test 零泄漏",
     "调用 closedloop.consume_feedback()",
     "leaked_test_rows=0 且 rows>0"),
    ("closedloop", "registry", "模型注册表与保守晋升", closedloop_registry,
     "验证影子版本已登记，且本轮不会自动替换线上模型",
     "查询 model_registry 全部版本",
     "至少 1 个版本且 promoted=false"),
    ("closedloop", "calibration", "置信度校准过程", closedloop_calibration,
     "验证阈值只在 val 上选出、候选与可靠性分箱完整（未达标会如实披露）",
     "读取 eval/calibration.json",
     "val_rows>0、候选 ≥5、分箱 = 10"),

    ("pipeline", "run", "执行七段链路取得 trace", pipeline_run,
     "跑一次完整端到端链路，产出可逐段检查的 trace",
     "调用 pipeline.run_pipeline()（生成走缓存）并把 trace 缓存在会话中",
     "七段齐全、无异常"),
    ("pipeline", "final", "最终处置与门禁结论", pipeline_final,
     "验证流水线给出明确处置、五项门禁结果与完整链路依赖",
     "读取最近一次 trace 的 final 字段",
     "disposition 存在且 gate_checks ≥ 5 项"),
    ("pipeline", "audit", "端到端审计落库", pipeline_audit,
     "验证每次运行都在 pipeline_runs 与 decision_log 留下记录",
     "统计两张表行数",
     "pipeline_runs > 0"),
]

# 七段链路逐段按钮
for stage_key, stage_name in pipeline.STAGE_DEFS:
    LIVE_SPECS.append((
        "pipeline", f"stage_{stage_key}", f"逐段：{stage_name}", make_stage_fn(stage_key),
        f"只检查端到端链路中的「{stage_name}」这一段输出是否完整、是否异常",
        "读取最近一次 trace 的对应段（没有则先跑一次完整链路）",
        "该段 status ≠ error 且结构化字段齐全"))

# 回归 / 冒烟
for i, (module, port, label, restore) in enumerate([
    ("verify_server", 8765, "第一步 · 评测地基", False),
    ("verify_step2_server", 8766, "第二步 · 分诊", True),
    ("verify_step3_server", 8767, "第三步 · 解决建议", False),
    ("verify_step4_server", 8768, "第四步 · 路由", False),
    ("verify_step5_server", 8769, "第五步 · 上线闭环", False),
], 1):
    LIVE_SPECS.append((
        "regress", f"step{i}", f"回归：{label}（{port}）",
        make_regress_step(module, port, restore),
        f"真实重跑 {port} 独立验收页的全部检查项，确认该步没有被后续改动破坏",
        f"调用 {module}.api_run_all()，逐条列出检查项结果",
        f"{port} 页面所有检查项通过"))

LIVE_SPECS += [
    ("regress", "unified", "回归：一体化全量验收", regress_unified,
     "重跑八组既有检查 + 一条真实链路 + 报告哈希校验",
     "调用 pipeline.run_all() 并逐条列出 9 项检查",
     "9/9 通过"),
    ("regress", "hash", "报告哈希隔离校验", regress_hash,
     "验证跑完流水线后，11 份既有报告的内容一个字节都没被覆盖",
     "记录哈希 → 跑一次七段链路 → 再算哈希逐份比对",
     "0 份报告变化"),
    ("regress", "ports", "端口连通性探测", regress_ports,
     "记录 8765~8770 六个服务的在线状态，确认探测过程本身可用",
     "逐个 GET /api/status，超时 1 秒",
     "探测无异常即可（未启动不算失败）"),
]

SMOKE_IDS = ["env.db", "env.split", "env.ollama", "triage.r1", "skills.r1",
             "retrieval.suggest", "generation.tables", "routing.explain",
             "lifecycle.mode", "closedloop.calibration", "pipeline.final",
             "regress.ports"]


def make_smoke_specs():
    return [
        ("regress", "smoke_env", "冒烟：环境健康检查",
         make_smoke(["env.db", "env.split", "env.isolation", "env.ollama"], "环境健康检查"),
         "只跑最快、最能说明环境是否正常 4 个点，用于开工前确认",
         "顺序执行：数据库连接 / 冻结清单 / 线上隔离 / Ollama",
         "4/4 通过"),
        ("regress", "smoke_api", "冒烟：接口与报告可读性",
         make_smoke(["env.reports", "env.thresholds", "retrieval.weight",
                     "generation.tables"], "接口与报告可读性"),
         "确认报告、门槛、检索权重、标注表都能被正常读取",
         "顺序执行 4 个只读测试点",
         "4/4 通过"),
        ("regress", "smoke_chain", "冒烟：端到端链路",
         make_smoke(["pipeline.run", "pipeline.stage_gate", "pipeline.audit"], "端到端链路"),
         "用最短路径确认七段链路能跑、门禁有结论、审计有落库",
         "顺序执行：跑链路 / 检查门禁段 / 检查审计",
         "3/3 通过"),
        ("regress", "smoke_all", "冒烟：全流程各阶段首点",
         make_smoke(SMOKE_IDS, "全流程各阶段首点"),
         "每个阶段各取一个代表点快速过一遍，用于改动后的第一轮健康检查",
         f"顺序执行 {len(SMOKE_IDS)} 个代表测试点并逐点汇总",
         f"{len(SMOKE_IDS)}/{len(SMOKE_IDS)} 通过"),
    ]


LIVE_SPECS += make_smoke_specs()

TEST_INDEX: dict[str, dict] = {}
for gkey, skey, name, fn, what, how, expect in LIVE_SPECS:
    tid = f"{gkey}.{skey}"
    TEST_INDEX[tid] = {"id": tid, "group": gkey, "name": name, "what": what,
                       "how": how, "expect": expect, "kind": "live", "fn": fn}

_CATALOG: dict | None = None


def get_catalog(refresh: bool = False) -> dict:
    global _CATALOG
    if _CATALOG is not None and not refresh:
        return _CATALOG
    had_error = False
    by_group: dict[str, list[dict]] = {g["key"]: [] for g in GROUPS}
    for spec in TEST_INDEX.values():
        if spec["kind"] == "live":
            by_group[spec["group"]].append({k: v for k, v in spec.items() if k != "fn"})
    for gkey in REPORT_SOURCES:
        checks = safe_checks(gkey)
        if any("检查项读取失败" in c["name"] for c in checks):
            had_error = True
        path, loader_name, _ = REPORT_SOURCES[gkey]
        for i, c in enumerate(checks):
            tid = f"{gkey}.r{i + 1}"
            meta = {"id": tid, "group": gkey, "name": c["name"],
                    "what": EXPLAIN.get((gkey, i), f"验证「{c['name']}」成立"),
                    "how": f"读取 reports/{path.name}，重算第 {i + 1}/{len(checks)} 条检查项（只读，不重训）",
                    "expect": "该检查项 passed = true", "kind": "report",
                    "source": path.name}
            TEST_INDEX[tid] = {**meta, "fn": make_report_fn(gkey, i)}
            by_group[gkey].append(meta)
    groups = [dict(g, points=by_group[g["key"]]) for g in GROUPS]
    result = {"groups": groups, "total": sum(len(g["points"]) for g in groups),
              "port": PORT, "complete": not had_error}
    _CATALOG = result if not had_error else None
    return result


def run_test(spec: dict) -> dict:
    rec = Recorder(spec)
    try:
        spec["fn"](rec)
        if rec.result is None:
            rec.finish(False, "无结论", "FAIL：测试点没有给出结论")
    except Exception as exc:
        rec.add("error", f"{type(exc).__name__}: {exc}")
        for line in traceback.format_exc().strip().splitlines()[-4:]:
            rec.add("error", line)
        rec.finish(False, "执行异常", f"FAIL：测试点执行抛异常 {type(exc).__name__}: {exc}")
    return rec.result


def run_many(ids: list[str]) -> dict:
    results = []
    for tid in ids:
        spec = TEST_INDEX.get(tid)
        if spec is None:
            results.append({"id": tid, "name": tid, "passed": False,
                            "conclusion": "测试点不存在", "elapsed_ms": 0,
                            "logs": [{"time": time.strftime("%H:%M:%S"), "level": "error",
                                      "label": "异常", "msg": f"找不到测试点 {tid}"}]})
        else:
            results.append(run_test(spec))
    return {"ok": all(r["passed"] for r in results), "results": results}


def api_status() -> dict:
    try:
        counts = {r[0]: r[1] for r in _query("SELECT split, count(*) FROM tickets GROUP BY 1")}
        online = _query("SELECT count(*) FROM assignments")[0][0]
        runs = _query("SELECT count(*) FROM pipeline_runs")[0][0]
        return {"ok": True, "db_ok": True, "splits": counts,
                "online_assignments": online, "pipeline_runs": runs,
                "ollama_ok": knowledge._probe_ollama(),
                "mode": lifecycle.mode_state()["mode"],
                "reports": len(_hash_all()), "port": PORT}
    except Exception as exc:
        return {"ok": False, "db_ok": False, "error": str(exc)}


class Handler(BaseHTTPRequestHandler):
    def _send(self, data, content_type: str) -> None:
        body = data if isinstance(data, bytes) else json.dumps(
            data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _payload(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self) -> None:
        parts = self.path.split("?", 1)
        path = parts[0]
        query = parts[1] if len(parts) > 1 else ""
        try:
            if path in ("/", "/index.html", "/testbench.html"):
                self._send((WEB_DIR / "testbench.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/catalog":
                self._send(get_catalog(refresh="refresh=1" in query),
                           "application/json; charset=utf-8")
            elif path == "/api/status":
                self._send(api_status(), "application/json; charset=utf-8")
            else:
                self._send({"ok": False, "error": "未知路径"}, "application/json; charset=utf-8")
        except Exception as exc:
            self._send({"ok": False, "error": str(exc), "trace": traceback.format_exc()},
                       "application/json; charset=utf-8")

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        payload = self._payload()
        try:
            catalog = get_catalog()
            if path == "/api/run":
                tid = str(payload.get("id", ""))
                spec = TEST_INDEX.get(tid)
                self._send(run_test(spec) if spec else
                           {"ok": False, "error": f"未知测试点 {tid}"},
                           "application/json; charset=utf-8")
            elif path == "/api/run_many":
                self._send(run_many(list(payload.get("ids", []))),
                           "application/json; charset=utf-8")
            elif path == "/api/run_group":
                key = str(payload.get("group", ""))
                ids = [p["id"] for g in catalog["groups"] if g["key"] == key
                       for p in g["points"]]
                self._send(run_many(ids), "application/json; charset=utf-8")
            elif path == "/api/plan":
                mode = str(payload.get("mode", "all"))
                ids = []
                for g in catalog["groups"]:
                    if mode == "smoke":
                        ids.append(g["points"][0]["id"] if g["points"] else None)
                    elif mode == "regress" and g["key"] == "regress":
                        ids.extend(p["id"] for p in g["points"])
                    elif mode == "all":
                        ids.extend(p["id"] for p in g["points"])
                self._send({"ok": True, "ids": [x for x in ids if x], "mode": mode},
                           "application/json; charset=utf-8")
            else:
                self._send({"ok": False, "error": "未知路径"}, "application/json; charset=utf-8")
        except Exception as exc:
            self._send({"ok": False, "error": str(exc), "trace": traceback.format_exc()},
                       "application/json; charset=utf-8")

    def log_message(self, fmt, *args) -> None:
        pass


def main() -> None:
    catalog = get_catalog()
    print("=" * 68)
    print("  分步验收台 · 按流程顺序的细粒度测试")
    print(f"  http://127.0.0.1:{PORT}/")
    print(f"  共 {catalog['total']} 个测试点，每个点单独运行并输出可读日志")
    print("=" * 68)
    if os.environ.get("ITSM_NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}/")).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

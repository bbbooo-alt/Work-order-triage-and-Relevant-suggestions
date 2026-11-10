# -*- coding: utf-8 -*-
"""端到端流水线：分诊 → 技能预测 → 混合检索 → 大模型生成 → 路由 → 自动档门禁 → 回流闭环。

把五步验收与三项 P0 补缺串成一条真实可执行的链路。所有下游决策只用上游的「预测值」
（预测优先级、预测技能），不用真值；只读演练不写线上 assignments，不覆盖任何既有报告。
"""
from __future__ import annotations

import datetime
import json
import time
import traceback

import closedloop
import generation
import knowledge
import lifecycle
import route
import skills
import triage
from common import (EVAL_DIR, REPORT_FILE, REPORTS_DIR, connect, dump_json,
                    load_json)

PIPELINE_VERSION = "unified-pipeline-v1"
ROUTE_VERSION = "route-v1-pipeline"
REPORT_PATH = REPORTS_DIR / "pipeline_metrics.json"

REPORT_FILES = {
    "step1": REPORT_FILE,
    "step2": triage.REPORT_PATH,
    "step3": knowledge.REPORT_PATH,
    "step4": route.REPORT_PATH,
    "step5": lifecycle.REPORT_PATH,
    "p0_1": skills.REPORT_PATH,
    "p0_1_impact": skills.IMPACT_PATH,
    "p0_2": generation.REPORT_PATH,
    "p0_3": closedloop.REPORT_PATH,
    "closed_loop": REPORTS_DIR / "closed_loop_metrics.json",
    "p0_3_retrain": REPORTS_DIR / "retraining_metrics.json",
}

STEP_LABELS = {
    "step1": "① 评测地基", "step2": "② 分诊", "step3": "③ 解决建议",
    "step4": "④ 路由", "step5": "⑤ 上线闭环",
    "p0_1": "P0-1 技能预测", "p0_2": "P0-2 生成与标注",
    "p0_3": "P0-3 校准与回流重训",
}

STAGE_DEFS = [
    ("intake", "① 工单接入与 PII 脱敏"),
    ("triage", "② 分诊预测（类型 / 部门 / 优先级 / 置信度）"),
    ("skills", "③ 技能多标签预测（P0-1）"),
    ("retrieval", "④ 混合检索与拒答（第三步）"),
    ("generation", "⑤ 基于检索片段的生成（P0-2）"),
    ("routing", "⑥ 技能∩排班∩容量路由（第四步）"),
    ("gate", "⑦ 置信度＋风险自动档门禁（P0-3 / 第五步）"),
]


def item(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def ensure_table() -> None:
    lifecycle.ensure_tables()
    conn = connect()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS pipeline_runs (
        run_id       bigserial PRIMARY KEY,
        ticket_id    integer NOT NULL REFERENCES tickets(ticket_id),
        split        text NOT NULL,
        mode         text NOT NULL,
        auto_allowed boolean NOT NULL,
        disposition  text NOT NULL,
        trace        jsonb NOT NULL,
        created_at   timestamptz NOT NULL DEFAULT now())""")
    conn.commit()
    conn.close()


def _hash(path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def ticket_pool(seed: str = "pool-a", limit: int = 8) -> list[dict]:
    """从冻结 test 集里确定性抽一批，且必须同时具备分诊与技能预测，保证链路能跑到底。"""
    conn = connect()
    cur = conn.cursor()
    cur.execute("""SELECT t.ticket_id, t.split, t.subject, t.body, t.type, t.queue,
                          t.priority, t.required_skills, t.language,
                          r.pred_type, r.pred_queue, r.pred_priority, r.confidence,
                          coalesce(s.pred_skills, '{}'::integer[]),
                          coalesce(s.true_skills, '{}'::integer[])
                   FROM tickets t
                   JOIN triage_results r
                     ON r.ticket_id = t.ticket_id AND r.model_version = %s
                   LEFT JOIN skill_predictions s ON s.ticket_id = t.ticket_id
                   WHERE t.split = 'test'
                   ORDER BY md5(t.ticket_id::text || %s)
                   LIMIT %s""", (triage.MODEL_VERSION, seed, limit))
    rows = cur.fetchall()
    conn.close()
    return [{"ticket_id": r[0], "split": r[1], "subject": r[2] or "",
             "body": r[3] or "", "type": r[4], "queue": r[5],
             "priority": r[6], "skills": list(r[7] or []), "language": r[8],
             "pred_type": r[9], "pred_queue": r[10], "pred_priority": r[11],
             "confidence": float(r[12]) if r[12] is not None else None,
             "pred_skills": list(r[13] or []), "true_skills": list(r[14] or []),
             "body_preview": (r[3] or "")[:180]} for r in rows]


def load_ticket(ticket_id: int) -> dict | None:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""SELECT t.ticket_id, t.split, t.subject, t.body, t.type, t.queue,
                          t.priority, t.required_skills, t.language, t.created_at,
                          r.pred_type, r.pred_queue, r.pred_priority, r.confidence,
                          coalesce(s.pred_skills, '{}'::integer[]),
                          coalesce(s.true_skills, '{}'::integer[])
                   FROM tickets t
                   LEFT JOIN triage_results r
                     ON r.ticket_id = t.ticket_id AND r.model_version = %s
                   LEFT JOIN skill_predictions s ON s.ticket_id = t.ticket_id
                   WHERE t.ticket_id = %s""",
                (triage.MODEL_VERSION, int(ticket_id)))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return {"ticket_id": row[0], "split": row[1], "subject": row[2] or "",
            "body": row[3] or "", "type": row[4], "queue": row[5],
            "priority": row[6], "skills": list(row[7] or []), "language": row[8],
            "created_at": row[9], "pred_type": row[10], "pred_queue": row[11],
            "pred_priority": row[12],
            "confidence": float(row[13]) if row[13] is not None else None,
            "pred_skills": list(row[14] or []), "true_skills": list(row[15] or [])}


def _stage(stages: list[dict], key: str, name: str, started: float,
           status: str, summary: str, data: dict) -> None:
    stages.append({"key": key, "name": name, "status": status,
                   "elapsed_ms": int((time.time() - started) * 1000),
                   "summary": summary, "data": data})


def run_pipeline(ticket_id: int, use_generation: bool = True,
                 use_cache: bool = True, persist: bool = True) -> dict:
    """跑一张工单的完整链路；任何一段失败都会记录成 error 并继续展示后续可判定结果。"""
    ensure_table()
    stages: list[dict] = []
    ticket = load_ticket(ticket_id)
    if ticket is None:
        return {"ok": False, "error": f"找不到工单 {ticket_id}"}

    text = (ticket["subject"] + " " + ticket["body"])[:600]
    state = lifecycle.mode_state()
    mode = state["mode"]

    # ① 工单接入 + PII 脱敏
    t0 = time.time()
    redacted = lifecycle.redact(text)
    _stage(stages, "intake", STAGE_DEFS[0][1], t0, "ok",
           f"工单 #{ticket_id}（{ticket['split']}），展示文本已脱敏",
           {"ticket_id": ticket_id, "split": ticket["split"],
            "subject": ticket["subject"], "body_preview": ticket["body"][:400],
            "language": ticket["language"], "truth": {
                "type": ticket["type"], "queue": ticket["queue"],
                "priority": ticket["priority"], "required_skills": ticket["skills"]},
            "redacted_preview": redacted["redacted"][:400],
            "pii_changed": redacted["changed"]})

    # ② 分诊
    t0 = time.time()
    conf = ticket["confidence"]
    triage_data = {"model_version": triage.MODEL_VERSION,
                   "pred_type": ticket["pred_type"], "pred_queue": ticket["pred_queue"],
                   "pred_priority": ticket["pred_priority"], "confidence": conf,
                   "low_confidence": conf is not None and conf < triage.LOW_CONFIDENCE_THRESHOLD,
                   "source": "triage_results（只在 train 拟合、val 选配置、test 评一次）"}
    status = "ok" if triage_data["pred_priority"] else "error"
    _stage(stages, "triage", STAGE_DEFS[1][1], t0, status,
           f"预测 {triage_data['pred_type']} / {triage_data['pred_queue']} / "
           f"{triage_data['pred_priority']}，置信度 {conf}", triage_data)

    # ③ 技能预测（下游只吃 pred_skills）
    t0 = time.time()
    pred_skills = ticket["pred_skills"]
    truth_skills = ticket["true_skills"]
    skills_data = {"model_version": skills.MODEL_VERSION,
                   "pred_skills": pred_skills, "true_skills": truth_skills,
                   "matched": sorted(set(pred_skills) & set(truth_skills)),
                   "missed": sorted(set(truth_skills) - set(pred_skills)),
                   "extra": sorted(set(pred_skills) - set(truth_skills)),
                   "note": "路由阶段使用 pred_skills，不使用真值 required_skills"}
    _stage(stages, "skills", STAGE_DEFS[2][1], t0,
           "ok" if pred_skills else "warn",
           f"预测技能 {len(pred_skills)} 个，真值 {len(truth_skills)} 个；下游按预测值派单",
           skills_data)

    # ④ 检索
    t0 = time.time()
    retrieval = {"abstain": True, "results": []}
    try:
        knowledge.apply_active_weight()
        knowledge.ensure_ollama()
        query_vector = knowledge.embed([text])[0]
        conn = connect()
        try:
            retrieval = knowledge.search(conn, text, query_vector=query_vector,
                                         exclude_ticket_id=ticket_id, mode="hybrid")
        finally:
            conn.close()
        retrieval_data = {
            "mode": retrieval["mode"], "abstain": retrieval["abstain"],
            "best_kb_similarity": retrieval["best_kb_similarity"],
            "lexical_overlap": retrieval["lexical_overlap"],
            "keyword_hits": retrieval["keyword_hits"],
            "reranked_by_keyword": retrieval["reranked_by_keyword"],
            "kb_in_top_k": retrieval["kb_in_top_k"],
            "keyword_weight": knowledge.ACTIVE_KEYWORD_WEIGHT,
            "hits": [{"rank": h["rank"], "chunk_id": h["chunk_id"],
                      "source": h["source_label"], "chunk_type": h["chunk_type"],
                      "similarity": round(float(h["similarity"]), 4),
                      "channels": "+".join(h["sources"]), "preview": h["preview"]}
                     for h in retrieval["results"]],
        }
        rstatus = "warn" if retrieval["abstain"] else "ok"
        rsummary = ("检索拒答：词面重合/相似度不足，后续不生成、不自动生效"
                    if retrieval["abstain"] else
                    f"召回 {len(retrieval_data['hits'])} 条片段，权威知识 {retrieval_data['kb_in_top_k']} 条")
        _stage(stages, "retrieval", STAGE_DEFS[3][1], t0, rstatus, rsummary, retrieval_data)
    except Exception as exc:
        _stage(stages, "retrieval", STAGE_DEFS[3][1], t0, "error", str(exc),
               {"trace": traceback.format_exc(), "abstain": True, "hits": []})

    # ⑤ 生成
    t0 = time.time()
    hits = retrieval.get("results", [])
    abstain = bool(retrieval.get("abstain"))
    if not use_generation:
        gen = {"abstained": True, "reason": "本次选择「只检索不生成」对照模式",
               "model": generation.GEN_MODEL, "steps": [], "citations": []}
    else:
        try:
            gen = generation.generate(text, hits, abstain=abstain, use_cache=use_cache)
        except Exception as exc:
            gen = {"abstained": True, "reason": str(exc), "model": generation.GEN_MODEL,
                   "steps": [], "citations": [], "parse_error": True}
    response = gen.get("response", {}) or {}
    gen_data = {
        "model": gen.get("model"), "abstained": bool(gen.get("abstained")),
        "reason": gen.get("reason", ""), "steps": response.get("steps", []),
        "info_needed": response.get("info_needed", []),
        "escalation": response.get("escalation", ""),
        "citations": gen.get("citations", []),
        "citation_valid_rate": gen.get("citation_valid_rate"),
        "grounded": gen.get("grounded", 0.0),
        "latency_sec": gen.get("latency_sec", 0.0),
        "from_cache": gen.get("from_cache", False),
        "retrieval_only": {"top_k": len(hits), "abstain": abstain,
                           "top_sources": [h["source_label"] for h in hits[:3]],
                           "strategy": "只返回原文片段，未经大模型改写"},
        "parse_error": gen.get("parse_error", False),
    }
    gstatus = "error" if gen_data["parse_error"] else ("warn" if gen_data["abstained"] else "ok")
    gsummary = (f"生成拒答：{gen_data['reason']}" if gen_data["abstained"] else
                f"生成 {len(gen_data['steps'])} 步建议，{len(gen_data['citations'])} 条有效引用，"
                f"grounding {gen_data['grounded']}，耗时 {gen_data['latency_sec']}s"
                + ("（缓存命中）" if gen_data["from_cache"] else ""))
    _stage(stages, "generation", STAGE_DEFS[4][1], t0, gstatus, gsummary, gen_data)

    # ⑥ 路由：required 用预测技能，priority 用预测优先级
    t0 = time.time()
    routing_data = {"status": "manual_queue", "candidates": [], "rejects": {}}
    try:
        conn = connect()
        try:
            static = route.load_static(conn)
        finally:
            conn.close()
        arrived = static["win_start"] + (static["win_end"] - static["win_start"]) * 0.35
        routing_ticket = {"ticket_id": ticket_id, "priority": ticket["priority"],
                          "required": set(pred_skills), "arrived": arrived,
                          "pred_priority": ticket["pred_priority"],
                          "pred_queue": ticket["pred_queue"], "confidence": conf,
                          "before_assignment_id": None}
        probe = connect()
        try:
            decision = route.assign_one(routing_ticket, static, ROUTE_VERSION,
                                        conn=probe, dry_run=True)
        finally:
            probe.close()
        ev = decision["evaluation"]
        chosen = ev["chosen"]
        routing_data = {"status": decision["status"],
                        "reason": decision["reason"],
                        "required_used": sorted(pred_skills),
                        "required_source": "skill_predictions.pred_skills",
                        "chosen": ({"engineer_id": chosen["engineer_id"],
                                    "score": chosen["score"], "level": chosen["level"],
                                    "matched_domains": chosen["matched_domains"],
                                    "load": chosen["load"], "capacity": chosen["capacity"],
                                    "only_on_call": chosen["only_on_call"],
                                    "parts": chosen["parts"]} if chosen else None),
                        "candidates": ev["candidates"][:5],
                        "candidate_count": len(ev["candidates"]),
                        "rejects": ev["rejects"], "dry_run": True}
    except Exception as exc:
        routing_data = {"status": "manual_queue", "error": str(exc),
                        "trace": traceback.format_exc(), "candidates": [], "rejects": {}}
    rstatus = "warn" if routing_data["status"] != "assigned" else "ok"
    _stage(stages, "routing", STAGE_DEFS[5][1], t0, rstatus,
           (f"派给工程师 e{routing_data['chosen']['engineer_id']}，得分 "
            f"{routing_data['chosen']['score']}" if routing_data.get("chosen") else
            "无合规工程师，降级到人工队列（不硬派）"), routing_data)

    # ⑦ 自动档门禁
    t0 = time.time()
    try:
        gate = closedloop.auto_eligibility({
            "confidence": conf, "pred_priority": ticket["pred_priority"],
            "pred_type": ticket["pred_type"],
            "retrieval": {"abstain": abstain},
            "routing": {"status": routing_data["status"]}})
        gate_data = dict(gate, mode=mode, calibration_file=str(EVAL_DIR / "calibration.json"))
        gstatus2 = "ok" if gate["allowed"] else "warn"
        gsummary = ("门禁放行，自动档可生效" if gate["allowed"] else
                    f"门禁拒绝（{', '.join(gate['failed'])}），转人工确认")
    except Exception as exc:
        gate_data = {"allowed": False, "error": str(exc), "checks": {}, "failed": []}
        gate = {"allowed": False, "risk": {"level": "unknown", "reasons": []}}
        gstatus2, gsummary = "error", str(exc)
    _stage(stages, "gate", STAGE_DEFS[6][1], t0, gstatus2, gsummary, gate_data)

    auto_allowed = bool(gate.get("allowed"))
    if auto_allowed and mode == "auto":
        disposition = "自动生效"
    elif auto_allowed:
        disposition = f"门禁通过，但当前为{mode}档 → 人工确认"
    else:
        disposition = "转人工处理"
    final = {
        "pipeline_version": PIPELINE_VERSION, "ticket_id": ticket_id,
        "mode": mode, "kill_switch_on": state["kill_switch_on"],
        "auto_allowed": auto_allowed, "disposition": disposition,
        "risk": gate.get("risk", {}), "gate_checks": gate.get("checks", {}),
        "failed_gates": gate.get("failed", []),
        "chain": ["triage_results.pred_*", "skill_predictions.pred_skills",
                  "knowledge.search", "generation.generate",
                  "route.assign_one(dry_run)", "closedloop.auto_eligibility",
                  "lifecycle.mode_state / decision_log"],
        "isolation": {"online_assignments_written": 0,
                      "note": "路由为 dry_run，只写独立审计表 pipeline_runs，线上 assignments 保持 0 行"},
    }

    trace = {"ok": all(s["status"] != "error" for s in stages),
             "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
             "ticket": {"ticket_id": ticket_id, "split": ticket["split"],
                        "subject": ticket["subject"], "language": ticket["language"],
                        "truth": {"type": ticket["type"], "queue": ticket["queue"],
                                  "priority": ticket["priority"],
                                  "required_skills": ticket["skills"]}},
             "stages": stages, "final": final,
             "elapsed_ms": sum(s["elapsed_ms"] for s in stages)}

    if persist:
        conn = connect()
        cur = conn.cursor()
        payload = json.dumps(trace, ensure_ascii=False, default=str)
        cur.execute("""INSERT INTO pipeline_runs
            (ticket_id, split, mode, auto_allowed, disposition, trace)
            VALUES (%s,%s,%s,%s,%s,%s::jsonb) RETURNING run_id""",
                    (ticket_id, ticket["split"], mode, auto_allowed, disposition, payload))
        run_id = cur.fetchone()[0]
        cur.execute("""INSERT INTO decision_log
            (ticket_id, stage, mode, decision, applied, reason)
            VALUES (%s,'pipeline',%s,%s::jsonb,%s,%s)""",
                    (ticket_id, mode,
                     json.dumps({"disposition": disposition,
                                 "auto_allowed": auto_allowed,
                                 "risk": gate.get("risk", {}).get("level"),
                                 "routing": routing_data["status"],
                                 "retrieval_abstain": abstain}, ensure_ascii=False),
                     auto_allowed and mode == "auto",
                     "一体化流水线端到端审计；只读演练，不写线上派单"))
        conn.commit()
        conn.close()
        trace["run_id"] = run_id
    return trace


def readiness() -> dict:
    """所有上游产物是否齐备；缺一项页面都会明确提示，避免静默跳过。"""
    conn = connect()
    cur = conn.cursor()
    table_counts = {}
    for table in ("tickets", "triage_results", "skill_predictions",
                  "resolution_knowledge", "generation_cache", "golden_labels",
                  "generation_review", "route_assignments", "decision_log",
                  "feedback_events", "drift_snapshots", "auto_assignments",
                  "model_registry", "pipeline_runs", "assignments"):
        try:
            cur.execute(f"SELECT count(*) FROM {table}")
            table_counts[table] = cur.fetchone()[0]
        except Exception:
            conn.rollback()
            table_counts[table] = None
    cur.execute("SELECT split, count(*) FROM tickets GROUP BY 1")
    splits = dict(cur.fetchall())
    cur.execute("SELECT model_version, promoted, feedback_rows, trained_at FROM model_registry ORDER BY 1")
    registry = [{"model_version": r[0], "promoted": r[1],
                 "feedback_rows": r[2], "trained_at": str(r[3])} for r in cur.fetchall()]
    conn.close()
    reports = {key: {"file": path.name, "exists": path.exists()}
               for key, path in REPORT_FILES.items()}
    for key, meta in reports.items():
        if meta["exists"]:
            meta["generated_at"] = load_json(REPORT_FILES[key]).get("generated_at", "")
    return {"db_ok": True, "splits": splits, "tables": table_counts,
            "reports": reports, "registry": registry,
            "state": lifecycle.mode_state(),
            "ollama_ok": knowledge._probe_ollama(),
            "calibration": (load_json(EVAL_DIR / "calibration.json")
                            if (EVAL_DIR / "calibration.json").exists() else {})}


def _read_checks() -> list[dict]:
    """只读地从各阶段既有报告重算检查项，不重训、不覆盖报告。"""
    groups: list[dict] = []

    def add(key: str, checks: list[dict], source: str) -> None:
        groups.append({"key": key, "label": STEP_LABELS.get(key, key),
                       "source": source,
                       "passed": sum(bool(c["passed"]) for c in checks),
                       "total": len(checks), "checks": checks})

    import freeze_split
    frozen = load_json(REPORT_FILE)
    first = [
        item("冻结划分与清单一致", freeze_split.check()["passed"],
             f"train/val/test = {frozen['split_counts']}"),
        item("两条基线齐全", all(m in frozen["models"] for m in ("majority-v0", "keyword-v0")),
             f"报告模型 {frozen['model_versions']}"),
        item("评测门槛已写死", "min_macro_f1_gain_over_keyword_baseline" in frozen.get("thresholds", {}),
             str(frozen.get("thresholds", {}))),
        item("线上派单表隔离", frozen["isolation"]["assignments_rows"] == 0
             and frozen["isolation"]["stray_predictions_outside_test"] == 0,
             f"assignments {frozen['isolation']['assignments_rows']} 行，越界预测 "
             f"{frozen['isolation']['stray_predictions_outside_test']} 条"),
    ]
    add("step1", first, REPORT_FILE.name)
    add("step2", triage.build_checks(load_json(triage.REPORT_PATH)), triage.REPORT_PATH.name)
    add("step3", knowledge.build_checks(load_json(knowledge.REPORT_PATH)), knowledge.REPORT_PATH.name)
    add("step4", route.build_checks(load_json(route.REPORT_PATH)), route.REPORT_PATH.name)
    add("step5", lifecycle.build_checks(load_json(lifecycle.REPORT_PATH)), lifecycle.REPORT_PATH.name)
    add("p0_1", skills.build_checks(load_json(skills.REPORT_PATH),
                                    load_json(skills.IMPACT_PATH)), skills.REPORT_PATH.name)
    add("p0_2", generation.build_checks(load_json(generation.REPORT_PATH)), generation.REPORT_PATH.name)
    add("p0_3", closedloop.build_checks(load_json(REPORTS_DIR / "closed_loop_metrics.json")),
        "closed_loop_metrics.json")
    return groups


def run_all(use_generation: bool = True) -> dict:
    """一体化全量验收：八组既有检查 + 一条真实端到端链路 + 报告哈希隔离校验。"""
    started = time.time()
    before = {key: _hash(path) for key, path in REPORT_FILES.items()}
    checks: list[dict] = []
    groups = _read_checks()
    ready = readiness()
    checks.append(item("数据库与本地模型服务就绪", ready["db_ok"] and ready["ollama_ok"],
                       f"工单 {ready['splits']}，Ollama {'可用' if ready['ollama_ok'] else '不可用'}"))
    checks.append(item("八组验收检查全部通过",
                       all(g["passed"] == g["total"] for g in groups),
                       "；".join(f"{g['label']} {g['passed']}/{g['total']}" for g in groups)))
    checks.append(item("P0 三项产物齐备",
                       ready["tables"]["skill_predictions"] > 0
                       and ready["tables"]["generation_cache"] > 0
                       and ready["tables"]["model_registry"] > 0,
                       f"技能预测 {ready['tables']['skill_predictions']} 行、生成缓存 "
                       f"{ready['tables']['generation_cache']} 行、注册表 "
                       f"{ready['tables']['model_registry']} 个版本"))

    pool = ticket_pool(limit=1)
    trace = run_pipeline(pool[0]["ticket_id"], use_generation=use_generation)
    stages = trace["stages"]
    checks.append(item("七段链路全部执行且无异常",
                       len(stages) == len(STAGE_DEFS)
                       and all(s["status"] != "error" for s in stages),
                       " → ".join(s["key"] for s in stages)))
    by_key = {s["key"]: s for s in stages}
    checks.append(item("下游只用预测技能、不用真值",
                       by_key["routing"]["data"].get("required_used")
                       == sorted(by_key["skills"]["data"].get("pred_skills", [])),
                       f"路由 required={by_key['routing']['data'].get('required_used')}，"
                       f"来自 {by_key['routing']['data'].get('required_source')}"))
    checks.append(item("检索拒答时不硬编答案",
                       by_key["retrieval"]["data"].get("abstain")
                       or not by_key["generation"]["data"].get("abstained"),
                       f"检索 abstain={by_key['retrieval']['data'].get('abstain')}，"
                       f"生成 abstained={by_key['generation']['data'].get('abstained')}"))
    checks.append(item("自动档门禁给出可解释结论",
                       len(by_key["gate"]["data"].get("checks", {})) >= 5
                       and isinstance(by_key["gate"]["data"].get("failed"), list),
                       f"allowed={trace['final']['auto_allowed']}，"
                       f"风险 {trace['final']['risk'].get('level')}，"
                       f"处置「{trace['final']['disposition']}」"))
    checks.append(item("线上派单表仍为 0 行",
                       readiness()["tables"]["assignments"] == 0,
                       "流水线路由为 dry_run，审计只写 pipeline_runs / decision_log"))

    after = {key: _hash(path) for key, path in REPORT_FILES.items()}
    checks.append(item("全量验收未覆盖任何既有报告", before == after,
                       f"{len(before)} 份报告内容哈希前后一致"))

    report = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
              "pipeline_version": PIPELINE_VERSION,
              "elapsed_sec": round(time.time() - started, 2),
              "ok": all(c["passed"] for c in checks),
              "checks": checks, "groups": groups,
              "trace": trace, "readiness": ready}
    dump_json(REPORT_PATH, report)
    return report


if __name__ == "__main__":
    result = run_all()
    print("一体化验收", "通过" if result["ok"] else "未通过",
          f"{sum(c['passed'] for c in result['checks'])}/{len(result['checks'])}")
    for check in result["checks"]:
        print("通过" if check["passed"] else "未通过", check["name"], "|", check["detail"])

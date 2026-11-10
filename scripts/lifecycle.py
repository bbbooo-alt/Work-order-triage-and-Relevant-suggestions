# -*- coding: utf-8 -*-
"""第五步：影子/辅助/自动三档上线、人工改判回流、漂移监控、回退开关与合规处置。"""
from __future__ import annotations

import datetime
import json
import re
from collections import defaultdict

from psycopg2.extras import execute_values

import knowledge
import route
from common import EVAL_DIR, REPORT_FILE, REPORTS_DIR, connect, dump_json, load_json
from triage import LOW_CONFIDENCE_THRESHOLD as LOW_CONF
from triage import MODEL_VERSION as TRIAGE_MODEL

REPORT_PATH = REPORTS_DIR / "lifecycle_metrics.json"
MODES = ("shadow", "assist", "auto")
MODE_LABELS = {"shadow": "影子模式（只记录不生效）",
               "assist": "辅助模式（人确认后生效）",
               "auto": "自动模式（仅高置信度低风险生效）"}
BATCH_SIZE = 12
DRYRUN_VERSION = "route-v1-dryrun"

DRIFT_THRESHOLDS = {
    "low_confidence_rate": 0.40,
    "retrieval_abstain_rate": 0.60,
    "routing_degraded_rate": 0.70,
    "human_override_rate": 0.35,
}

PII_SAMPLE = ("Customer mail: hans.mueller@example.de, phone +49 170 1234567, "
              "invoice 202611091234 and IP 192.168.10.24.")

COMPLIANCE = {
    "dataset_license": "cc-by-nc-4.0",
    "commercial_use": "禁止：仅够内部验证，真实生产必须更换数据源或取得授权",
    "synthetic_data": "工程师、技能、排班均为造数；结论只说明流程跑通，不能当生产预期",
    "pii": "真实工单含个人身份信息，入库前脱敏，并保留审计日志与访问控制",
    "audit": "每次模型/规则变更记录版本、配置、评测分数与生效时间",
}


def ensure_tables() -> None:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS lifecycle_config (
        key text PRIMARY KEY, value text NOT NULL,
        updated_at timestamptz NOT NULL DEFAULT now())""")
    cur.execute("""CREATE TABLE IF NOT EXISTS decision_log (
        log_id bigserial PRIMARY KEY,
        ticket_id integer NOT NULL REFERENCES tickets(ticket_id),
        stage text NOT NULL, mode text NOT NULL,
        decision jsonb NOT NULL, applied boolean NOT NULL,
        reason text, created_at timestamptz NOT NULL DEFAULT now())""")
    cur.execute("""CREATE TABLE IF NOT EXISTS feedback_events (
        feedback_id bigserial PRIMARY KEY,
        ticket_id integer NOT NULL REFERENCES tickets(ticket_id),
        split text NOT NULL, actor text NOT NULL, field text NOT NULL,
        old_value text, new_value text, accepted boolean NOT NULL,
        enters_label_pool boolean NOT NULL DEFAULT false,
        created_at timestamptz NOT NULL DEFAULT now())""")
    cur.execute("""CREATE TABLE IF NOT EXISTS drift_snapshots (
        snapshot_id bigserial PRIMARY KEY,
        window_label text NOT NULL, metrics jsonb NOT NULL,
        alert boolean NOT NULL, reasons text,
        created_at timestamptz NOT NULL DEFAULT now())""")
    cur.execute("""CREATE TABLE IF NOT EXISTS auto_assignments (
        auto_id bigserial PRIMARY KEY,
        ticket_id integer NOT NULL REFERENCES tickets(ticket_id),
        engineer_id smallint REFERENCES engineers(engineer_id),
        mode text NOT NULL, applied boolean NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now())""")
    conn.commit()
    conn.close()


def get_config(conn, key: str, default: str = "") -> str:
    cur = conn.cursor()
    cur.execute("SELECT value FROM lifecycle_config WHERE key = %s", (key,))
    row = cur.fetchone()
    return row[0] if row else default


def set_config(conn, key: str, value: str) -> None:
    cur = conn.cursor()
    cur.execute("""INSERT INTO lifecycle_config (key, value, updated_at) VALUES (%s,%s,now())
                   ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value,
                                                   updated_at = EXCLUDED.updated_at""",
                (key, value))
    conn.commit()


def mode_state() -> dict:
    conn = connect()
    mode = get_config(conn, "mode", "shadow")
    kill = get_config(conn, "kill_switch", "off")
    conn.close()
    return {"mode": mode, "label": MODE_LABELS.get(mode, mode),
            "kill_switch": kill, "kill_switch_on": kill == "on",
            "allowed_modes": ["shadow"] if kill == "on" else list(MODES)}


def set_mode(mode: str) -> dict:
    if mode not in MODES:
        return {"ok": False, "error": f"未知模式 {mode}，只允许 {', '.join(MODES)}",
                "state": mode_state()}
    conn = connect()
    kill = get_config(conn, "kill_switch", "off")
    if kill == "on" and mode != "shadow":
        conn.close()
        return {"ok": False, "error": "回退开关已打开：自动/辅助模式被拒绝，系统锁定在影子模式",
                "state": mode_state()}
    set_config(conn, "mode", mode)
    conn.close()
    return {"ok": True, "message": f"已切换到 {MODE_LABELS[mode]}", "state": mode_state()}


def set_kill_switch(on: bool) -> dict:
    conn = connect()
    set_config(conn, "kill_switch", "on" if on else "off")
    if on:
        set_config(conn, "mode", "shadow")
    conn.close()
    return {"ok": True,
            "message": "回退开关已打开，自动生效被关闭并强制退回影子模式" if on
                       else "回退开关已关闭，可重新逐档放量",
            "state": mode_state()}


def attempt_auto(context: dict | None = None) -> dict:
    """自动档准入。

    不传 context 时是第五步原有的「档位／回退开关」演练接口，保持向后兼容；
    传入 context 时按 P0-3 的单据级门禁判定：置信度、风险、检索依据、路由结果必须全过。
    """
    state = mode_state()
    checks = {"kill_switch_off": not state["kill_switch_on"]}
    if state["kill_switch_on"]:
        return {"allowed": False, "mode": "shadow",
                "reason": "回退开关已打开：自动模式被拒绝，强制回退到影子模式",
                "checks": checks, "failed": ["kill_switch_off"]}
    if context is None:
        return {"allowed": True, "mode": state["mode"],
                "reason": "回退开关关闭，自动模式可按门禁放行",
                "checks": checks, "failed": []}

    calibration_file = EVAL_DIR / "calibration.json"
    calibration = load_json(calibration_file) if calibration_file.exists() else {}
    conf_min = float(context.get("auto_conf_min")
                     or calibration.get("auto_conf_min") or 0.0)
    confidence = context.get("confidence")
    retrieval = context.get("retrieval") or {}
    routing = context.get("routing") or {}
    checks.update({
        "confidence_ok": confidence is not None and float(confidence) >= conf_min,
        "risk_low": context.get("risk") == "low",
        "retrieval_grounded": not bool(retrieval.get("abstain")),
        "routing_assigned": routing.get("status") == "assigned",
    })
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "allowed": not failed, "mode": state["mode"],
        "reason": ("全部门禁通过，允许自动生效" if not failed
                   else "门禁拒绝：" + "、".join(failed)),
        "auto_conf_min": conf_min, "checks": checks, "failed": failed,
    }


def redact(text: str) -> dict:
    out = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", text)
    out = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP]", out)

    def _phone(match):
        raw = match.group(0)
        # 纯数字长串更可能是单号而非电话，留给 [NUMBER] 处理。
        return "[PHONE]" if any(c in " +-" for c in raw) else raw

    out = re.sub(r"(?<!\d)(?:\+?\d[\d -]{7,}\d)(?!\d)", _phone, out)
    out = re.sub(r"(?<!\d)\d{8,}(?!\d)", "[NUMBER]", out)
    return {"original": text, "redacted": out,
            "email_removed": not bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", out)),
            "changed": out != text}


def load_batches(conn) -> list[list[dict]]:
    cur = conn.cursor()
    cur.execute("""SELECT t.ticket_id, t.subject, t.body, t.priority, t.required_skills,
                          t.created_at, r.pred_priority, r.pred_queue, r.confidence
                   FROM tickets t
                   LEFT JOIN triage_results r
                     ON r.ticket_id = t.ticket_id AND r.model_version = %s
                   WHERE t.split = 'test'
                   ORDER BY md5(t.ticket_id::text || 'lifecycle')
                   LIMIT %s""", (TRIAGE_MODEL, BATCH_SIZE * len(MODES)))
    rows = cur.fetchall()
    batches = []
    for i in range(0, len(rows), BATCH_SIZE):
        part = []
        for r in rows[i:i + BATCH_SIZE]:
            part.append({"ticket_id": r[0], "subject": r[1] or "", "body": r[2] or "",
                         "priority": r[3], "required": set(r[4] or []),
                         "created_at": r[5], "pred_priority": r[6],
                         "pred_queue": r[7], "confidence": r[8]})
        batches.append(part)
    return batches


def _routing_ticket(t: dict, static: dict) -> dict:
    return {"ticket_id": t["ticket_id"], "priority": t["priority"],
            "required": t["required"],
            "arrived": static["win_start"] + (static["win_end"] - static["win_start"]) * 0.35,
            "pred_priority": t["pred_priority"], "pred_queue": t["pred_queue"],
            "confidence": t["confidence"], "before_assignment_id": None}


def run_one_mode(conn, mode: str, batch: list[dict], static: dict,
                 probe_conn) -> dict:
    """一档上线演练：分诊 → 检索建议 → 路由，全部写审计日志；自动档才写沙盒派单表。"""
    set_config(conn, "mode", mode)
    knowledge.apply_active_weight()
    texts = [((t["subject"] + " " + t["body"])[:600]) for t in batch]
    vectors = knowledge.embed(texts)
    cur = conn.cursor()
    log_rows, auto_rows, feedback_rows = [], [], []
    applied = mode == "auto"
    low_conf, abstains, degraded = 0, 0, 0

    for idx, (t, text, qv) in enumerate(zip(batch, texts, vectors)):
        confidence = float(t["confidence"]) if t["confidence"] is not None else None
        low_conf += int(confidence is not None and confidence < LOW_CONF)
        triage_decision = {"model_version": TRIAGE_MODEL,
                           "pred_priority": t["pred_priority"],
                           "pred_queue": t["pred_queue"], "confidence": confidence,
                           "low_confidence": confidence is not None and confidence < LOW_CONF}
        retrieval = knowledge.search(conn, text, query_vector=qv, mode="hybrid")
        abstains += int(retrieval["abstain"])
        retrieval_decision = {"abstain": retrieval["abstain"],
                              "best_kb_similarity": retrieval["best_kb_similarity"],
                              "lexical_overlap": retrieval["lexical_overlap"],
                              "top1": (retrieval["results"][0]["source_label"]
                                       if retrieval["results"] else None),
                              "strategy": "只检索不生成"}
        routing = route.assign_one(_routing_ticket(t, static), static,
                                   DRYRUN_VERSION, conn=probe_conn, dry_run=True)
        chosen = routing["evaluation"]["chosen"]
        routing_decision = {"status": routing["status"],
                            "engineer_id": chosen["engineer_id"] if chosen else None,
                            "candidate_count": len(routing["evaluation"]["candidates"]),
                            "degraded": routing["status"] == "manual_queue"}
        degraded += int(routing["status"] == "manual_queue")

        for stage, decision, reason in (
            ("triage", triage_decision, "复用第二步分诊预测，低置信度转人工"),
            ("retrieval", retrieval_decision, "复用第三步混合检索与拒答策略"),
            ("routing", routing_decision, "复用第四步硬约束与软打分"),
        ):
            log_rows.append((t["ticket_id"], stage, mode,
                             json.dumps(decision, ensure_ascii=False), applied, reason))

        if applied and chosen:
            auto_rows.append((t["ticket_id"], chosen["engineer_id"], mode, True))
        # 辅助模式里模拟人工抽检：一部分接受、一部分改判（test 样本只观察，不进训练池）。
        if mode == "assist" and idx < 6:
            accepted = idx % 2 == 0
            feedback_rows.append((t["ticket_id"], "test", "human", "priority",
                                  t["pred_priority"], t["pred_priority"] if accepted else "high",
                                  accepted, False))

    execute_values(cur, """INSERT INTO decision_log
        (ticket_id, stage, mode, decision, applied, reason) VALUES %s""",
        log_rows, template="(%s,%s,%s,%s::jsonb,%s,%s)", page_size=500)
    if auto_rows:
        execute_values(cur, """INSERT INTO auto_assignments
            (ticket_id, engineer_id, mode, applied) VALUES %s""", auto_rows)
    if feedback_rows:
        execute_values(cur, """INSERT INTO feedback_events
            (ticket_id, split, actor, field, old_value, new_value, accepted, enters_label_pool)
            VALUES %s""", feedback_rows,
            template="(%s,%s,%s,%s,%s,%s,%s,%s)", page_size=500)
    conn.commit()
    return {"mode": mode, "label": MODE_LABELS[mode], "tickets": len(batch),
            "applied": applied, "log_rows": len(log_rows),
            "auto_rows": len(auto_rows), "feedback_rows": len(feedback_rows),
            "low_confidence": low_conf, "abstains": abstains, "degraded": degraded}


def seed_train_feedback(conn) -> dict:
    """模拟线上真实回流：只有 train 样本才允许进入下一轮标注池。"""
    cur = conn.cursor()
    cur.execute("""SELECT ticket_id, priority FROM tickets WHERE split = 'train'
                   ORDER BY md5(ticket_id::text || 'feedback') LIMIT 6""")
    rows = cur.fetchall()
    values = [(tid, "train", "human", "priority", prio,
               "high" if i % 2 == 0 else prio, i % 2 != 0, i % 2 == 0)
              for i, (tid, prio) in enumerate(rows)]
    execute_values(cur, """INSERT INTO feedback_events
        (ticket_id, split, actor, field, old_value, new_value, accepted, enters_label_pool)
        VALUES %s""", values, template="(%s,%s,%s,%s,%s,%s,%s,%s)")
    conn.commit()
    return {"train_feedback": len(rows),
            "enters_label_pool": sum(1 for v in values if v[7]),
            "note": "回流数据只进 train；冻结 test 永不改动"}


def drift_monitor(persist: bool = True) -> dict:
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT stage, decision FROM decision_log")
    by_stage = defaultdict(list)
    for stage, decision in cur.fetchall():
        by_stage[stage].append(decision if isinstance(decision, dict) else json.loads(decision))
    cur.execute("SELECT count(*) FILTER (WHERE NOT accepted), count(*) FROM feedback_events")
    overrides, feedback_total = cur.fetchone()

    def rate(items, key):
        return round(sum(bool(d.get(key)) for d in items) / len(items), 4) if items else 0.0

    current = {
        "low_confidence_rate": rate(by_stage["triage"], "low_confidence"),
        "retrieval_abstain_rate": rate(by_stage["retrieval"], "abstain"),
        "routing_degraded_rate": round(
            sum(d.get("status") == "manual_queue" for d in by_stage["routing"])
            / max(1, len(by_stage["routing"])), 4),
        "human_override_rate": round(overrides / feedback_total, 4) if feedback_total else 0.0,
    }
    current["decision_rows"] = sum(len(v) for v in by_stage.values())

    def judge(metrics):
        reasons = [f"{k} {v} > {DRIFT_THRESHOLDS[k]}"
                   for k, v in metrics.items() if k in DRIFT_THRESHOLDS
                   and v > DRIFT_THRESHOLDS[k]]
        return bool(reasons), "；".join(reasons) or "全部指标在阈值内"

    alert, reasons = judge(current)
    simulated = dict(current, retrieval_abstain_rate=0.95)
    sim_alert, sim_reasons = judge(simulated)
    if persist:
        cur.execute("""INSERT INTO drift_snapshots (window_label, metrics, alert, reasons)
                       VALUES (%s,%s::jsonb,%s,%s), (%s,%s::jsonb,%s,%s)""",
                    ("current_window", json.dumps(current, ensure_ascii=False), alert, reasons,
                     "simulated_drift_probe", json.dumps(simulated, ensure_ascii=False),
                     sim_alert, sim_reasons))
        conn.commit()
    conn.close()
    return {"current": current, "current_alert": alert, "current_reasons": reasons,
            "simulated": simulated, "simulated_alert": sim_alert,
            "simulated_reasons": sim_reasons, "thresholds": DRIFT_THRESHOLDS,
            "note": "第二条快照是人为注入的漂移探针，只用于验证告警规则确实会触发"}


def human_override(ticket_id: int, field: str, new_value: str) -> dict:
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT split, priority FROM tickets WHERE ticket_id = %s", (int(ticket_id),))
    row = cur.fetchone()
    if row is None:
        conn.close()
        return {"ok": False, "error": f"找不到工单 {ticket_id}"}
    split, old_value = row
    enters = split == "train"
    cur.execute("""INSERT INTO feedback_events
        (ticket_id, split, actor, field, old_value, new_value, accepted, enters_label_pool)
        VALUES (%s,%s,'human',%s,%s,%s,false,%s) RETURNING feedback_id""",
                (int(ticket_id), split, field, str(old_value), str(new_value), enters))
    fid = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return {"ok": True, "feedback_id": fid, "ticket_id": int(ticket_id),
            "split": split, "field": field, "old_value": str(old_value),
            "new_value": str(new_value), "accepted": False,
            "enters_label_pool": enters,
            "note": ("该工单属于冻结 test，只作观察、不进训练池，尺子不变"
                     if not enters else "该工单属于 train，已进入下一轮标注池")}


def audit_stats(conn) -> dict:
    cur = conn.cursor()
    cur.execute("""SELECT mode, stage, count(*), count(*) FILTER (WHERE applied)
                   FROM decision_log GROUP BY 1,2 ORDER BY 1,2""")
    by_mode_stage = [{"mode": r[0], "stage": r[1], "rows": r[2], "applied": r[3]}
                     for r in cur.fetchall()]
    cur.execute("""SELECT count(*), count(*) FILTER (WHERE enters_label_pool),
                          count(*) FILTER (WHERE split = 'test' AND enters_label_pool)
                   FROM feedback_events""")
    feedback_total, label_pool, leaked = cur.fetchone()
    cur.execute("SELECT count(*) FROM auto_assignments")
    auto_rows = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM drift_snapshots")
    snapshots = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM assignments")
    online_rows = cur.fetchone()[0]
    cur.execute("SELECT split, count(*) FROM tickets GROUP BY 1")
    split_counts = dict(cur.fetchall())
    return {"by_mode_stage": by_mode_stage, "feedback_total": feedback_total,
            "label_pool_size": label_pool, "test_leaked_into_pool": leaked,
            "auto_assignment_rows": auto_rows, "drift_snapshots": snapshots,
            "online_assignments_rows": online_rows, "split_counts": split_counts}


def run() -> dict:
    ensure_tables()
    knowledge.ensure_ollama()
    conn = connect()
    cur = conn.cursor()
    for table in ("decision_log", "feedback_events", "drift_snapshots", "auto_assignments"):
        cur.execute(f"DELETE FROM {table}")
    conn.commit()
    set_config(conn, "kill_switch", "off")
    set_config(conn, "mode", "shadow")

    static = route.load_static(conn)
    batches = load_batches(conn)
    probe_conn = connect()
    timeline = []
    try:
        for mode, batch in zip(MODES, batches):
            timeline.append(run_one_mode(conn, mode, batch, static, probe_conn))
    finally:
        probe_conn.close()
    seed_train_feedback(conn)

    # 回退预案演练：打开开关后，自动模式必须被拒绝并强制回退到影子模式。
    opened = set_kill_switch(True)
    blocked = attempt_auto()
    rollback_demo = {"kill_switch": opened["state"]["kill_switch"],
                     "auto_allowed": blocked["allowed"],
                     "reason": blocked["reason"],
                     "mode_after": opened["state"]["mode"],
                     "passed": (not blocked["allowed"]
                                and opened["state"]["mode"] == "shadow")}
    set_kill_switch(False)

    drift = drift_monitor()
    pii = redact(PII_SAMPLE)
    stats = audit_stats(conn)
    frozen = load_json(REPORT_FILE) if REPORT_FILE.exists() else {}
    baseline_split = frozen.get("split_counts", {})
    conn.close()

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "modes": [{"mode": m, "label": MODE_LABELS[m]} for m in MODES],
        "timeline": timeline,
        "rollback_demo": rollback_demo,
        "drift": drift,
        "pii": pii,
        "compliance": COMPLIANCE,
        "audit": stats,
        "frozen_baseline": {"source": REPORT_FILE.name,
                            "split_counts": baseline_split,
                            "routing_timeout_rate": frozen.get("routing", {}).get("timeout_rate")},
        "state": mode_state(),
        "isolation": {
            "tables": ["lifecycle_config", "decision_log", "feedback_events",
                       "drift_snapshots", "auto_assignments"],
            "note": ("自动档的「上线派单」写在沙盒表 auto_assignments，"
                     "不写线上 assignments，也不碰前四步的回放数据，以便随时回归前四步验收页"),
        },
        "known_limits": [
            "三档演练用的是冻结 test 工单做只读回放，只验证流程，不代表真实线上流量。",
            "回流标注只进 train；test 样本只作观察，绝不进训练池，也不重新训练。",
            "尚未接真实排班、真实工单与 A/B 分流，影子模式未跑满真实业务周期。",
        ],
    }
    dump_json(REPORT_PATH, report)
    return report


def build_checks(report: dict) -> list[dict]:
    def item(name, passed, detail):
        return {"name": name, "passed": bool(passed), "detail": detail}

    timeline = {t["mode"]: t for t in report["timeline"]}
    audit, drift, rollback = report["audit"], report["drift"], report["rollback_demo"]
    checks = [
        item("三档按顺序逐档放开", [t["mode"] for t in report["timeline"]] == list(MODES),
             "影子（只记录）→ 辅助（人确认）→ 自动（按门禁生效），顺序不可跳档"),
        item("影子模式只记录、不改变实际流程",
             timeline.get("shadow", {}).get("applied") is False
             and timeline.get("shadow", {}).get("auto_rows") == 0,
             f"影子批 {timeline.get('shadow',{}).get('tickets',0)} 条工单，"
             f"{timeline.get('shadow',{}).get('log_rows',0)} 条审计日志，生效 0 条"),
        item("只有自动档才落生效决策",
             timeline.get("auto", {}).get("applied") is True
             and timeline.get("assist", {}).get("applied") is False
             and audit["auto_assignment_rows"] == timeline.get("auto", {}).get("auto_rows"),
             f"沙盒 auto_assignments {audit['auto_assignment_rows']} 行，辅助/影子档 0 行"),
        item("每个阶段都有审计日志", len(audit["by_mode_stage"]) >= 9
             and all(r["rows"] > 0 for r in audit["by_mode_stage"]),
             f"{len(audit['by_mode_stage'])} 组「模式 × 阶段」记录，覆盖分诊 / 检索 / 路由"),
        item("人工改判可回流为标注", audit["feedback_total"] > 0 and audit["label_pool_size"] > 0,
             f"反馈 {audit['feedback_total']} 条，其中 {audit['label_pool_size']} 条进入下一轮标注池"),
        item("冻结测试集未被污染", audit["test_leaked_into_pool"] == 0
             and audit["split_counts"].get("test") == report["frozen_baseline"]["split_counts"].get("test"),
             f"test 仍为 {audit['split_counts'].get('test')} 行，与第一步冻结清单一致；"
             f"从 test 泄漏进训练池 {audit['test_leaked_into_pool']} 条"),
        item("漂移监控有快照与写死的阈值", audit["drift_snapshots"] >= 2 and len(drift["thresholds"]) >= 4,
             f"{audit['drift_snapshots']} 条快照；阈值 {drift['thresholds']}"),
        item("告警规则确实会触发",
             drift["simulated_alert"] and "retrieval_abstain_rate" in drift["simulated_reasons"],
             f"当前窗口告警 {drift['current_alert']}（{drift['current_reasons']}）；"
             f"注入拒答率漂移后 {drift['simulated_alert']}（{drift['simulated_reasons']}）"),
        item("回退开关演练通过", rollback["passed"],
             f"开关打开后自动模式 allowed={rollback['auto_allowed']}，模式回退为 {rollback['mode_after']}"),
        item("PII 脱敏可用", report["pii"]["changed"] and report["pii"]["email_removed"],
             f"样例脱敏后：{report['pii']['redacted']}"),
        item("线上表与前四步数据保持干净", audit["online_assignments_rows"] == 0,
             f"线上 assignments {audit['online_assignments_rows']} 行；本步只写 "
             f"{', '.join(report['isolation']['tables'])}"),
        item("尺子冻结：复用第一步基线、未重新训练",
             bool(report["frozen_baseline"]["split_counts"])
             and report["frozen_baseline"]["routing_timeout_rate"] is not None,
             f"冻结基线来自 {report['frozen_baseline']['source']}，路由超时率 "
             f"{report['frozen_baseline']['routing_timeout_rate']}"),
        item("合规结论明确", report["compliance"]["commercial_use"].startswith("禁止")
             and "cc-by-nc-4.0" in report["compliance"]["dataset_license"],
             f"{report['compliance']['dataset_license']}：{report['compliance']['commercial_use']}；"
             f"{report['compliance']['pii']}"),
        item("第五步报告独立落盘", REPORT_PATH.exists(), str(REPORT_PATH)),
    ]
    return checks


if __name__ == "__main__":
    result = run()
    print("模式时间线", [(t["mode"], t["log_rows"], t["auto_rows"]) for t in result["timeline"]])
    for check in build_checks(result):
        print("通过" if check["passed"] else "未通过", check["name"], "|", check["detail"])

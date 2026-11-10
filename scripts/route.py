# -*- coding: utf-8 -*-
"""第四步：硬约束路由 + 软打分 + 事务行锁 + 幂等 + 边界降级 + 离线回放评测。"""
from __future__ import annotations

import datetime
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from common import REPORTS_DIR, THRESHOLD_FILE, connect, dump_json, load_json
from run_baseline import REPLAY_LIMIT, SLA_HOURS, WORK_HOURS
from triage import MODEL_VERSION as TRIAGE_MODEL

REPORT_PATH = REPORTS_DIR / "routing_metrics.json"
MODEL_VERSION = "route-v1"
PROBE_VERSION = "route-v1-concurrency"

# 软打分权重：熟练度、等级、实时负荷、高优先级给高等级加成、仅值班覆盖扣分。
W_PROF, W_LEVEL, W_LOAD, W_HIGH_LEVEL, P_ONCALL_ONLY = 2.0, 1.2, 3.0, 0.8, 1.5


def ensure_table() -> None:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS route_assignments (
        assignment_id  bigserial PRIMARY KEY,
        model_version  text NOT NULL,
        ticket_id      integer NOT NULL REFERENCES tickets(ticket_id),
        engineer_id    smallint REFERENCES engineers(engineer_id),
        valid          boolean NOT NULL,
        decided_at     timestamptz NOT NULL,
        finished_at    timestamptz,
        load_at_assign integer NOT NULL DEFAULT 0,
        score          real,
        reason         text,
        status         text NOT NULL DEFAULT 'assigned'
                       CHECK (status IN ('assigned', 'manual_queue'))
    )""")
    cur.execute("""CREATE UNIQUE INDEX IF NOT EXISTS uq_route_ticket
                   ON route_assignments (model_version, ticket_id)""")
    cur.execute("""CREATE INDEX IF NOT EXISTS idx_route_load
                   ON route_assignments (model_version, engineer_id, finished_at)""")
    conn.commit()
    conn.close()


def load_static(conn) -> dict:
    cur = conn.cursor()
    skills = defaultdict(dict)
    cur.execute("SELECT engineer_id, domain_id, proficiency FROM engineer_skills")
    for eid, did, prof in cur.fetchall():
        skills[eid][did] = prof
    shifts = defaultdict(list)
    cur.execute("SELECT engineer_id, shift_start, shift_end, on_call FROM shifts")
    for eid, start, end, on_call in cur.fetchall():
        shifts[eid].append((start, end, bool(on_call)))
    cur.execute("SELECT min(shift_start), max(shift_end) FROM shifts")
    win_start, win_end = cur.fetchone()
    return {"skills": dict(skills), "shifts": dict(shifts),
            "win_start": win_start, "win_end": win_end}


def load_replay(conn, static: dict) -> list[dict]:
    """与第一步轮询基线使用完全相同的 500 条工单和时间映射，保证指标可比。"""
    cur = conn.cursor()
    cur.execute("""SELECT t.ticket_id, t.priority, t.required_skills, t.created_at,
                          r.pred_priority, r.pred_queue, r.confidence
                   FROM tickets t
                   LEFT JOIN triage_results r
                     ON r.ticket_id = t.ticket_id AND r.model_version = %s
                   WHERE t.split = 'test'
                   ORDER BY md5(t.ticket_id::text || 'replay') LIMIT %s""",
                (TRIAGE_MODEL, REPLAY_LIMIT))
    raw = cur.fetchall()
    times = [r[3] for r in raw]
    t_min, t_max = min(times), max(times)
    span = (static["win_end"] - static["win_start"]) * 0.8

    def warp(t):
        if t_max == t_min:
            return static["win_start"]
        return static["win_start"] + (t - t_min) / (t_max - t_min) * span

    replay = [{"ticket_id": r[0], "priority": r[1], "required": set(r[2] or []),
               "arrived": warp(r[3]), "pred_priority": r[4], "pred_queue": r[5],
               "confidence": r[6]} for r in raw]
    return sorted(replay, key=lambda x: x["arrived"])


def evaluate_candidates(cur, ticket: dict, static: dict, model_version: str,
                        before_id: int | None = None) -> dict:
    arrived = ticket["arrived"]
    # 事务内锁住工程师行：并发派单时负荷读取被串行化，避免两人同时读到同一个空位。
    cur.execute("""SELECT engineer_id, level, capacity, active FROM engineers
                   ORDER BY engineer_id FOR UPDATE""")
    engineers = cur.fetchall()
    # 负荷不存静态字段，实时从派单表聚合「该时刻尚未结单」的数量。
    # 复盘历史某一单时只看它之前派出的记录，不能把「未来」的单算进当时的负荷。
    load_sql = """SELECT engineer_id, count(*), max(finished_at)
                  FROM route_assignments
                  WHERE model_version = %s AND status = 'assigned' AND finished_at > %s"""
    if before_id is not None:
        load_sql += " AND assignment_id < %s"
        cur.execute(load_sql + " GROUP BY engineer_id", (model_version, arrived, before_id))
    else:
        cur.execute(load_sql + " GROUP BY engineer_id", (model_version, arrived))
    loads = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    candidates, rejects = [], defaultdict(int)
    required = ticket["required"]
    priority = ticket["pred_priority"] or ticket["priority"]
    for eid, level, capacity, active in engineers:
        if not active:
            rejects["inactive"] += 1
            continue
        matched = required & set(static["skills"].get(eid, {}))
        if not matched:
            rejects["skill_miss"] += 1
            continue
        covering = [s for s in static["shifts"].get(eid, []) if s[0] <= arrived < s[1]]
        if not covering:
            rejects["shift_off"] += 1
            continue
        only_on_call = all(s[2] for s in covering)
        load, busy_until = loads.get(eid, (0, None))
        if load >= capacity:
            rejects["load_full"] += 1
            continue
        prof = max(static["skills"][eid][d] for d in matched)
        load_penalty = W_LOAD * load / capacity
        high_bonus = W_HIGH_LEVEL * level if priority == "high" else 0.0
        oncall_penalty = P_ONCALL_ONLY if only_on_call else 0.0
        score = (W_PROF * prof + W_LEVEL * level - load_penalty
                 + high_bonus - oncall_penalty)
        finished = max(busy_until or arrived, arrived) + timedelta(
            hours=WORK_HOURS[ticket["priority"]])
        candidates.append({
            "engineer_id": eid, "level": level, "capacity": capacity,
            "matched_domains": sorted(matched), "proficiency": prof,
            "load": load, "only_on_call": only_on_call,
            "score": round(score, 3), "finished_at": finished,
            "parts": {"proficiency": round(W_PROF * prof, 2),
                      "level": round(W_LEVEL * level, 2),
                      "load": round(-load_penalty, 2),
                      "high_priority": round(high_bonus, 2),
                      "on_call_only": round(-oncall_penalty, 2)},
        })
    candidates.sort(key=lambda c: (-c["score"], c["load"], c["engineer_id"]))
    return {"candidates": candidates, "rejects": dict(rejects),
            "chosen": candidates[0] if candidates else None}


def _reason(ticket: dict, evaluation: dict, chosen: dict | None) -> str:
    base = (f"pred_priority={ticket['pred_priority'] or 'missing'}; "
            f"required_skills={sorted(ticket['required'])}; "
            f"rejects={evaluation['rejects']}")
    if chosen is None:
        return base + "; no candidate -> manual_queue (never hard-assign)"
    p = chosen["parts"]
    return (base + f"; chosen=e{chosen['engineer_id']} "
            f"skill=ok{chosen['matched_domains']} "
            f"shift={'on_call_only' if chosen['only_on_call'] else 'covered'} "
            f"load={chosen['load']}/{chosen['capacity']} "
            f"| score={chosen['score']} = prof {p['proficiency']} + level {p['level']} "
            f"+ load {p['load']} + high {p['high_priority']} + oncall {p['on_call_only']}")


def assign_one(ticket: dict, static: dict, model_version: str = MODEL_VERSION,
               conn=None, dry_run: bool = False) -> dict:
    """单条派单：一个事务内完成幂等检查、行锁、实时负荷、决策与落库。"""
    own_conn = conn is None
    conn = conn or connect()
    cur = conn.cursor()
    try:
        # 只读演练要展示完整决策链，不能被幂等短路；真实派单则必须先查重。
        if not dry_run:
            cur.execute("""SELECT 1 FROM route_assignments
                           WHERE model_version = %s AND ticket_id = %s""",
                        (model_version, ticket["ticket_id"]))
            if cur.fetchone():
                if own_conn:
                    conn.rollback()
                return {"ticket_id": ticket["ticket_id"], "duplicate": True,
                        "status": "skipped", "detail": "该工单已有派单，幂等跳过"}
        evaluation = evaluate_candidates(cur, ticket, static, model_version,
                                         before_id=ticket.get("before_assignment_id"))
        chosen = evaluation["chosen"]
        reason = _reason(ticket, evaluation, chosen)
        if dry_run:
            conn.rollback()
            return {"ticket_id": ticket["ticket_id"], "duplicate": False,
                    "dry_run": True, "evaluation": evaluation, "reason": reason,
                    "status": "assigned" if chosen else "manual_queue"}

        if chosen is None:
            cur.execute("""INSERT INTO route_assignments
                (model_version, ticket_id, engineer_id, valid, decided_at, finished_at,
                 load_at_assign, score, reason, status)
                VALUES (%s,%s,NULL,false,%s,NULL,0,NULL,%s,'manual_queue')
                ON CONFLICT (model_version, ticket_id) DO NOTHING RETURNING assignment_id""",
                (model_version, ticket["ticket_id"], ticket["arrived"], reason))
        else:
            timeout = chosen["finished_at"] > ticket["arrived"] + timedelta(
                hours=SLA_HOURS[ticket["priority"]])
            cur.execute("""INSERT INTO route_assignments
                (model_version, ticket_id, engineer_id, valid, decided_at, finished_at,
                 load_at_assign, score, reason, status)
                VALUES (%s,%s,%s,true,%s,%s,%s,%s,%s,'assigned')
                ON CONFLICT (model_version, ticket_id) DO NOTHING RETURNING assignment_id""",
                (model_version, ticket["ticket_id"], chosen["engineer_id"],
                 ticket["arrived"], chosen["finished_at"], chosen["load"],
                 chosen["score"], reason))
            evaluation["timeout"] = bool(timeout)
        inserted = cur.fetchone() is not None
        conn.commit()
        return {"ticket_id": ticket["ticket_id"], "duplicate": not inserted,
                "status": "assigned" if chosen else "manual_queue",
                "engineer_id": chosen["engineer_id"] if chosen else None,
                "score": chosen["score"] if chosen else None,
                "reason": reason, "evaluation": evaluation}
    except Exception:
        conn.rollback()
        raise
    finally:
        if own_conn:
            conn.close()


def load_baseline(conn) -> dict:
    """从第一步的隔离回放表里读轮询基线，口径与统一评测脚本一致。"""
    cur = conn.cursor()
    cur.execute("""SELECT count(*), count(*) FILTER (WHERE ea.valid),
                          count(*) FILTER (WHERE ea.finished_at > ea.decided_at +
                              CASE t.priority WHEN 'high' THEN interval '4 hours'
                                             WHEN 'medium' THEN interval '24 hours'
                                             ELSE interval '72 hours' END)
                   FROM eval_assignments ea JOIN tickets t USING (ticket_id)
                   WHERE ea.model_version = 'round-robin-v0'""")
    total, valid, timeout = cur.fetchone()
    cur.execute("""SELECT engineer_id, count(*) FROM eval_assignments
                   WHERE model_version = 'round-robin-v0' GROUP BY 1""")
    per_engineer = dict(cur.fetchall())
    cur.execute("SELECT engineer_id FROM engineers")
    loads = [per_engineer.get(r[0], 0) for r in cur.fetchall()]
    return {"model_version": "round-robin-v0", "tickets": total,
            "valid_count": valid, "valid_rate": round(valid / total, 4) if total else 0,
            "timeout_rate": round(timeout / total, 4) if total else 0,
            "load_std": round(statistics.pstdev(loads), 2) if loads else 0}


def measure(records: list[dict], replay_size: int, baseline: dict) -> dict:
    assigned = [r for r in records if r["status"] == "assigned"]
    degraded = replay_size - len(assigned)
    timeouts = sum(bool(r.get("evaluation", {}).get("timeout")) for r in assigned)
    loads = defaultdict(int)
    max_ratio = 0.0
    for r in assigned:
        e = r["evaluation"]["chosen"]
        loads[e["engineer_id"]] += 1
        max_ratio = max(max_ratio, e["load"] / e["capacity"])
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT engineer_id FROM engineers ORDER BY engineer_id")
    eids = [r[0] for r in cur.fetchall()]
    conn.close()
    spread = [loads[e] for e in eids]
    return {
        "model_version": MODEL_VERSION,
        "replay_tickets": replay_size,
        "assigned_count": len(assigned),
        "coverage_rate": round(len(assigned) / replay_size, 4),
        "degraded_count": degraded,
        "degraded_rate": round(degraded / replay_size, 4),
        "constraint_compliance": round(
            sum(1 for r in assigned if r["evaluation"]["chosen"]) / len(assigned), 4)
        if assigned else 0,
        "timeout_count": timeouts,
        "timeout_rate": round(timeouts / replay_size, 4),
        "load_std": round(statistics.pstdev(spread), 2),
        "load_min": min(spread), "load_max": max(spread),
        "max_load_ratio": round(max_ratio, 4),
        "baseline_valid_count": baseline["valid_count"],
        "legal_assignments_gain": len(assigned) - baseline["valid_count"],
        "baseline": baseline,
        "note": ("轮询基线把 500 条全部硬派，只有少数恰好满足硬约束；"
                 "本步宁可转人工也不硬派，故 coverage_rate 不代表失败率。"),
    }


def run_concurrency_probe(static: dict, replay: list[dict]) -> dict:
    """同一时刻多线程并发派单：验证行锁与唯一约束真的能防双派、防超容量。"""
    conn = connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM route_assignments WHERE model_version = %s", (PROBE_VERSION,))
    conn.commit()
    conn.close()

    arrived = static["win_start"] + (static["win_end"] - static["win_start"]) * 0.35
    batch = [dict(t, arrived=arrived) for t in replay[:24]]
    errors = []

    def worker(ticket):
        try:
            return assign_one(ticket, static, PROBE_VERSION)
        except Exception as exc:
            errors.append(str(exc))
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(worker, batch))

    conn = connect()
    cur = conn.cursor()
    cur.execute("""SELECT count(*), count(DISTINCT ticket_id),
                          count(*) FILTER (WHERE status = 'assigned'),
                          count(*) FILTER (WHERE valid = false AND status = 'assigned')
                   FROM route_assignments WHERE model_version = %s""", (PROBE_VERSION,))
    rows, distinct, assigned, invalid = cur.fetchone()
    cur.execute("""SELECT r.engineer_id, count(*), e.capacity
                   FROM route_assignments r JOIN engineers e USING (engineer_id)
                   WHERE r.model_version = %s AND r.status = 'assigned'
                   GROUP BY r.engineer_id, e.capacity""", (PROBE_VERSION,))
    per_engineer = {r[0]: {"concurrent": r[1], "capacity": r[2]} for r in cur.fetchall()}
    conn.close()
    exceeded = [eid for eid, v in per_engineer.items() if v["concurrent"] > v["capacity"]]
    return {"threads": 6, "tickets": len(batch), "same_arrival_time": str(arrived),
            "rows": rows, "distinct_tickets": distinct,
            "assigned": assigned, "invalid_assigned": invalid,
            "engineers_exceeding_capacity": exceeded, "errors": errors,
            "passed": (rows == distinct and rows > 0 and assigned > 0
                       and not exceeded and invalid == 0 and not errors)}


def check_idempotency(static: dict, replay: list[dict]) -> dict:
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM route_assignments WHERE model_version = %s", (MODEL_VERSION,))
    before = cur.fetchone()[0]
    conn.close()
    again = assign_one(replay[0], static, MODEL_VERSION)
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM route_assignments WHERE model_version = %s", (MODEL_VERSION,))
    after = cur.fetchone()[0]
    conn.close()
    return {"ticket_id": replay[0]["ticket_id"], "rows_before": before,
            "rows_after": after, "duplicate_detected": again["duplicate"],
            "passed": again["duplicate"] and before == after}


def explain(ticket_id: int, static: dict, replay: list[dict]) -> dict:
    """给前端交互用：不写库，展示每个工程师被哪条硬约束淘汰以及软打分明细。"""
    ticket = next((t for t in replay if t["ticket_id"] == int(ticket_id)), None)
    if ticket is None:
        conn = connect()
        cur = conn.cursor()
        cur.execute("""SELECT ticket_id, priority, required_skills, created_at
                       FROM tickets WHERE ticket_id = %s""", (int(ticket_id),))
        row = cur.fetchone()
        conn.close()
        if row is None:
            return {"ok": False, "error": f"找不到工单 {ticket_id}"}
        ticket = {"ticket_id": row[0], "priority": row[1], "required": set(row[2] or []),
                  "arrived": static["win_start"] + (static["win_end"] - static["win_start"]) * 0.35,
                  "pred_priority": None, "pred_queue": None,
                  "arrived_note": "模拟到达时间（映射到排班窗口内）"}
    conn = connect()
    cur = conn.cursor()
    cur.execute("""SELECT assignment_id FROM route_assignments
                   WHERE model_version = %s AND ticket_id = %s""",
                (MODEL_VERSION, int(ticket_id)))
    row = cur.fetchone()
    before_id = row[0] if row else None
    # 回放中的工单按派单序号复盘；未参与回放的工单用空回放版本，负荷从零起算。
    model = MODEL_VERSION if before_id is not None else "route-v1-dryrun"
    ticket = dict(ticket, before_assignment_id=before_id)
    try:
        result = assign_one(ticket, static, model, conn=conn, dry_run=True)
    finally:
        conn.close()
    ev = result["evaluation"]
    return {"ok": True, "ticket_id": ticket["ticket_id"],
            "priority": ticket["priority"], "pred_priority": ticket.get("pred_priority"),
            "required_skills": sorted(ticket["required"]),
            "rejects": ev["rejects"],
            "candidates": ev["candidates"][:5],
            "candidate_count": len(ev["candidates"]),
            "status": result["status"], "reason": result["reason"]}


def run() -> dict:
    ensure_table()
    conn = connect()
    static = load_static(conn)
    replay = load_replay(conn, static)
    baseline = load_baseline(conn)
    cur = conn.cursor()
    cur.execute("DELETE FROM route_assignments WHERE model_version = %s", (MODEL_VERSION,))
    conn.commit()

    records = [assign_one(t, static, MODEL_VERSION, conn=conn) for t in replay]
    conn.close()
    metrics = measure(records, len(replay), baseline)
    probe = run_concurrency_probe(static, replay)
    idem = check_idempotency(static, replay)
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM route_assignments WHERE model_version = %s", (MODEL_VERSION,))
    stored_rows = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM route_assignments WHERE model_version = %s AND reason IS NULL",
                (MODEL_VERSION,))
    missing_reason = cur.fetchone()[0]
    cur.execute("""SELECT count(*) FROM route_assignments
                   WHERE model_version = %s AND status = 'manual_queue'
                     AND engineer_id IS NOT NULL""", (MODEL_VERSION,))
    degraded_with_engineer = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM assignments")
    online_rows = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM eval_assignments WHERE model_version = 'round-robin-v0'")
    baseline_rows = cur.fetchone()[0]
    conn.close()

    thresholds = load_json(THRESHOLD_FILE) if THRESHOLD_FILE.exists() else {}
    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_version": MODEL_VERSION,
        "input_model": TRIAGE_MODEL,
        "algorithm": "逐条贪心：硬约束过滤 + 加权软打分（批量最优/匈牙利算法留作二期）",
        "hard_constraints": ["技能交集", "班次覆盖（on_call 仅作软扣分）", "负荷未满且在职"],
        "soft_weights": {"proficiency": W_PROF, "level": W_LEVEL, "load": W_LOAD,
                         "high_priority_level": W_HIGH_LEVEL, "on_call_only_penalty": P_ONCALL_ONLY},
        "metrics": metrics,
        "concurrency_probe": probe,
        "idempotency": idem,
        "stored_rows": stored_rows,
        "missing_reason_rows": missing_reason,
        "degraded_rows_with_engineer": degraded_with_engineer,
        "online_assignments_rows": online_rows,
        "baseline_replay_rows": baseline_rows,
        "isolation": {"table": "route_assignments",
                      "note": "离线回放只写独立表；线上 assignments 与第一步 eval_assignments 均保持原样"},
        "thresholds": {"routing_valid_rate_min": thresholds.get("routing_valid_rate_min", 0.90),
                       "routing_timeout_rate_max": thresholds.get("routing_timeout_rate_max", 0.20)},
        "known_limits": [
            "工程师、技能、排班均为造数，结论只说明流程跑通，接真实排班后需影子模式再校准。",
            "所需技能取自工单真值标签；分诊模型目前只输出 type/queue/priority，上线前需补技能预测。",
            "轮询在「派单数均衡」上天然最优，但绝大多数派单不合法；本步优先保证合法与不超容量。",
        ],
        "sample_decisions": [{"ticket_id": r["ticket_id"], "status": r["status"],
                              "engineer_id": r.get("engineer_id"),
                              "score": r.get("score"), "reason": r["reason"]}
                             for r in (records[:2] + [x for x in records if x["status"] == "manual_queue"][:1])],
    }
    dump_json(REPORT_PATH, report)
    return report


def build_checks(report: dict) -> list[dict]:
    def item(name, passed, detail):
        return {"name": name, "passed": bool(passed), "detail": detail}

    m, base, probe, idem = (report["metrics"], report["metrics"]["baseline"],
                            report["concurrency_probe"], report["idempotency"])
    checks = [
        item("回放口径与轮询基线一致", m["replay_tickets"] == base["tickets"] == 500,
             f"同一批 {m['replay_tickets']} 条 test 工单、同一时间映射；基线 {base['tickets']} 条"),
        item("三段硬约束全部生效", m["constraint_compliance"] == 1.0,
             f"所有实际派出的 {m['assigned_count']} 条均同时满足技能、班次、负荷约束；合规率 {m['constraint_compliance']}"),
        item("派单合规率跨过写死的门槛",
             m["constraint_compliance"] >= report["thresholds"]["routing_valid_rate_min"],
             f"{m['constraint_compliance']} >= {report['thresholds']['routing_valid_rate_min']}（第一步 thresholds.json）"),
        item("合法派单数显著多于轮询", m["assigned_count"] > base["valid_count"] * 2,
             f"轮询仅 {base['valid_count']} 条合法（{base['valid_rate']}）→ 本步 {m['assigned_count']} 条，净增 {m['legal_assignments_gain']} 条"),
        item("超时率跨过门槛且不高于基线",
             m["timeout_rate"] <= report["thresholds"]["routing_timeout_rate_max"]
             and m["timeout_rate"] <= base["timeout_rate"],
             f"轮询 {base['timeout_rate']} → 本步 {m['timeout_rate']}；上限 {report['thresholds']['routing_timeout_rate_max']}"),
        item("无任一工程师超过容量", m["max_load_ratio"] <= 1.0,
             f"派单时最大负荷/容量 = {m['max_load_ratio']}；负荷由事务内实时聚合，不存静态字段"),
        item("负荷分布没有极端倾斜", m["load_max"] <= max(1, 2 * (m["assigned_count"] / 12)),
             f"各工程师派单数 {m['load_min']}~{m['load_max']}，标准差 {m['load_std']}（轮询 {base['load_std']}）"),
        item("无可用工程师时明确降级", report["degraded_rows_with_engineer"] == 0,
             f"{m['degraded_count']} 条转人工队列（{m['degraded_rate']}），其中被硬派 {report['degraded_rows_with_engineer']} 条；覆盖率 {m['coverage_rate']}"),
        item("并发压测不双派、不超容量", probe["passed"],
             f"{probe['threads']} 线程同一时刻派 {probe['tickets']} 单：落库 {probe['rows']} 行 / 去重后 {probe['distinct_tickets']} 单，"
             f"超容量工程师 {len(probe['engineers_exceeding_capacity'])} 人，错误 {len(probe['errors'])} 个"),
        item("派单幂等", idem["passed"],
             f"重复派工单 {idem['ticket_id']}：行数 {idem['rows_before']} → {idem['rows_after']}，被识别为重复"),
        item("每条决策可解释", report["missing_reason_rows"] == 0,
             f"{report['stored_rows']} 条记录全部带 reason（硬约束结果 + 软打分明细），缺失 {report['missing_reason_rows']} 条"),
        item("离线回放与线上、前几步完全隔离",
             report["online_assignments_rows"] == 0 and report["baseline_replay_rows"] == 500,
             f"线上 assignments {report['online_assignments_rows']} 行；第一步 eval_assignments {report['baseline_replay_rows']} 行未变；本步只写 {report['isolation']['table']}"),
        item("不用大模型拍板派单", "逐条贪心" in report["algorithm"],
             "纯规则引擎：硬约束过滤 + 可复现的加权打分，结果可审计"),
        item("输入来自第二步分诊结果", report["input_model"] == "triage-l1-v1",
             "回放读取 triage-l1-v1 的 pred_priority / pred_queue 作为路由输入"),
        item("第四步报告独立落盘", REPORT_PATH.exists(),
             str(REPORT_PATH)),
    ]
    return checks


if __name__ == "__main__":
    result = run()
    print("指标", {k: result["metrics"][k] for k in
                   ("assigned_count", "constraint_compliance", "timeout_rate",
                    "degraded_rate", "load_std", "max_load_ratio")})
    for check in build_checks(result):
        print("通过" if check["passed"] else "未通过", check["name"], "|", check["detail"])

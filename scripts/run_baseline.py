# -*- coding: utf-8 -*-
"""第一步基线：多数类、关键词规则分类，以及轮询派单的离线回放。可重复执行。"""
from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import timedelta

from psycopg2.extras import execute_values

from common import BASELINE_MODELS, ROUTING_MODEL, connect

REPLAY_LIMIT = 500
SLA_HOURS = {"high": 4, "medium": 24, "low": 72}
WORK_HOURS = {"high": 1, "medium": 3, "low": 6}

QUEUE_RULES = [
    ("Billing and Payments", ["billing", "invoice", "payment", "refund", "subscription", "rechnung", "zahlung", "abrechnung", "erstattung"]),
    ("Returns and Exchanges", ["return", "exchange", "warranty", "rücksendung", "rückgabe", "umtausch"]),
    ("Service Outages and Maintenance", ["outage", "maintenance", "downtime", "service disruption", "wartung", "ausfall", "störung"]),
    ("Human Resources", ["employee", "hr ", "onboarding", "payroll", "mitarbeiter", "personal"]),
    ("Sales and Pre-Sales", ["pricing", "quote", "marketing", "investment", "angebot", "verkauf"]),
    ("Technical Support", ["network", "vpn", "printer", "driver", "hardware", "software", "bug", "error", "router", "netzwerk", "treiber", "fehler"]),
    ("IT Support", ["server", "kubernetes", "ci/cd", "saas", "cloud", "infrastructure", "serverraum", "infrastruktur"]),
    ("Product Support", ["product", "feature", "integration", "compatibility", "produkt", "funktion", "kompatibilität"]),
]
TYPE_RULES = [
    ("Incident", ["outage", "failure", "not working", "down", "offline", "broken", "ausfall", "störung", "funktioniert nicht"]),
    ("Change", ["upgrade", "modify", "configuration change", "update request", "änderung", "anpassung", "aktualisierung"]),
    ("Problem", ["recurring", "intermittent", "root cause", "wiederholt", "zeitweise", "ursache"]),
]
PRIORITY_RULES = [
    ("high", ["urgent", "critical", "immediately", "asap", "severe", "emergency", "dringend", "kritisch", "sofort", "eilt"]),
    ("low", ["inquiry", "information", "documentation", "when convenient", "anfrage", "informationen", "dokumentation"]),
]


def predict_by_keywords(subject: str, body: str) -> tuple[str, str, str]:
    text = ((subject or "") + " " + (body or "")[:1200]).lower()

    def hit(keywords) -> bool:
        return any(k in text for k in keywords)

    queue = "Customer Service"
    for name, keywords in QUEUE_RULES:
        if hit(keywords):
            queue = name
            break
    typ = "Request"
    for name, keywords in TYPE_RULES:
        if hit(keywords):
            typ = name
            break
    priority = "medium"
    for name, keywords in PRIORITY_RULES:
        if hit(keywords):
            priority = name
            break
    return typ, queue, priority


def run() -> dict:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS eval_assignments (
            assignment_id bigserial PRIMARY KEY,
            model_version text NOT NULL,
            ticket_id     integer NOT NULL REFERENCES tickets(ticket_id),
            engineer_id   smallint NOT NULL REFERENCES engineers(engineer_id),
            valid         boolean NOT NULL,
            decided_at    timestamptz NOT NULL,
            finished_at   timestamptz NOT NULL,
            load_at_assign integer NOT NULL,
            reason        text
        )
    """)

    cur.execute("SELECT count(*) FROM tickets WHERE split = 'train'")
    train_total = cur.fetchone()[0]
    majority, confidence = {}, {}
    for column in ("type", "queue", "priority"):
        cur.execute(f"SELECT {column}, count(*) FROM tickets WHERE split = 'train' GROUP BY {column} ORDER BY 2 DESC LIMIT 1")
        value, cnt = cur.fetchone()
        majority[column] = value
        confidence[column] = round(cnt / train_total, 4)

    cur.execute("""SELECT ticket_id, subject, body, language, type, queue, priority, required_skills, created_at
                   FROM tickets WHERE split = 'test' ORDER BY ticket_id""")
    test_rows = cur.fetchall()

    cur.execute("DELETE FROM triage_results WHERE model_version = ANY(%s)", (BASELINE_MODELS,))
    predictions = []
    for tid, subject, body, lang, *_ in test_rows:
        typ, queue, prio = predict_by_keywords(subject, body)
        predictions.append((tid, "majority-v0", majority["type"], majority["queue"],
                            majority["priority"], confidence["priority"], "{}"))
        predictions.append((tid, "keyword-v0", typ, queue, prio, 0.6, "{}"))
    execute_values(cur, """INSERT INTO triage_results
        (ticket_id, model_version, pred_type, pred_queue, pred_priority, confidence, pred_skills)
        VALUES %s""", predictions, page_size=1000)

    # ---- 路由基线：轮询 + 离线回放（与线上 assignments 完全隔离）----
    cur.execute("DELETE FROM eval_assignments WHERE model_version = %s", (ROUTING_MODEL,))
    cur.execute("SELECT engineer_id, capacity FROM engineers WHERE active ORDER BY engineer_id")
    engineers = cur.fetchall()
    eids = [e[0] for e in engineers]
    capacities = dict(engineers)
    skills = defaultdict(set)
    cur.execute("SELECT engineer_id, domain_id FROM engineer_skills")
    for eid, did in cur.fetchall():
        skills[eid].add(did)
    shifts = defaultdict(list)
    cur.execute("SELECT engineer_id, shift_start, shift_end FROM shifts")
    for eid, start, end in cur.fetchall():
        shifts[eid].append((start, end))
    cur.execute("SELECT min(shift_start), max(shift_end) FROM shifts")
    win_start, win_end = cur.fetchone()
    span = (win_end - win_start) * 0.8

    cur.execute("""SELECT ticket_id, priority, required_skills, created_at
                   FROM tickets WHERE split = 'test'
                   ORDER BY md5(ticket_id::text || 'replay') LIMIT %s""", (REPLAY_LIMIT,))
    replay_raw = cur.fetchall()
    times = [r[3] for r in replay_raw]
    t_min, t_max = min(times), max(times)

    def warp(t):
        if t_max == t_min:
            return win_start
        return win_start + (t - t_min) / (t_max - t_min) * span

    replay = sorted(replay_raw, key=lambda r: warp(r[3]))
    finish_times = defaultdict(list)
    counts = defaultdict(int)
    records, valid_count, timeout_count = [], 0, 0
    for i, (tid, priority, required, created) in enumerate(replay):
        arrived = warp(created)
        eid = eids[i % len(eids)]
        finish_times[eid] = [t for t in finish_times[eid] if t > arrived]
        load = len(finish_times[eid])
        on_shift = any(s <= arrived < e for s, e in shifts[eid])
        skill_ok = bool(skills[eid] & set(required))
        load_ok = load < capacities[eid]
        valid = on_shift and skill_ok and load_ok
        busy_until = max(finish_times[eid], default=arrived)
        finished = busy_until + timedelta(hours=WORK_HOURS[priority])
        finish_times[eid].append(finished)
        counts[eid] += 1
        valid_count += int(valid)
        timeout_count += int(finished > arrived + timedelta(hours=SLA_HOURS[priority]))
        reason = f"round-robin; skill={'ok' if skill_ok else 'miss'} shift={'ok' if on_shift else 'off'} load={load}"
        records.append((ROUTING_MODEL, tid, eid, valid, arrived, finished, load, reason))
    execute_values(cur, """INSERT INTO eval_assignments
        (model_version, ticket_id, engineer_id, valid, decided_at, finished_at, load_at_assign, reason)
        VALUES %s""", records, page_size=1000)
    conn.commit()

    loads = [counts[e] for e in eids]
    result = {
        "test_tickets": len(test_rows),
        "prediction_rows": len(predictions),
        "models": BASELINE_MODELS,
        "majority": majority,
        "majority_confidence": confidence,
        "replay": {
            "model_version": ROUTING_MODEL,
            "tickets": len(replay),
            "valid_rate": round(valid_count / len(replay), 4),
            "timeout_rate": round(timeout_count / len(replay), 4),
            "load_std": round(statistics.pstdev(loads), 2),
            "load_min": min(loads),
            "load_max": max(loads),
            "isolated_table": "eval_assignments",
        },
    }
    conn.close()
    return result


if __name__ == "__main__":
    print(run())

# -*- coding: utf-8 -*-
"""按流程执行：建库 -> 建表 -> 清洗/造数 -> 导入 -> 建索引 -> 校验。可重复执行。"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv"
OUT = ROOT / "data/processed"
SQL_DIR = ROOT / "sql"

DB_NAME = os.environ.get("ITSM_DB", "itsm_triage")
PG_CONF = dict(
    host=os.environ.get("PGHOST", "127.0.0.1"),
    port=int(os.environ.get("PGPORT", "5432")),
    user=os.environ.get("PGUSER", "postgres"),
)
SEED = 20261107
TZ = timezone(timedelta(hours=8))
BASE_TIME = datetime(2026, 11, 7, 0, 0, 0, tzinfo=TZ)

GENERAL = "General Support"
DOMAIN_RULES = [
    ("Security & Compliance", ["security", "breach", "compliance", "privacy", "unauthorized", "phishing", "encryption", "hipaa", "gdpr", "vulnerability", "ciso", "soc", "alert"]),
    ("Network & Connectivity", ["network", "connectivity", "vpn", "wifi", "wlan", "bluetooth", "router", "switch", "repeater", "dns", "bandwidth", "ethernet", "connection", "firmware"]),
    ("Hardware & Devices", ["hardware", "driver", "printer", "audio", "video", "projector", "ssd", "nas", "headset", "mouse", "keyboard", "device", "equipment", "sensor", "overheating", "cpu", "motherboard", "scanner"]),
    ("Cloud & Platform", ["cloud", "saas", "kubernetes", "microservice", "ci/cd", "pipeline", "api", "integration", "orchestration", "deployment", "devops", "platform", "data lake", "analytics", "automation", "big data", "ml", "data"]),
    ("Software & Bugs", ["bug", "crash", "software", "performance", "compatibility", "error", "update", "patch", "code", "script", "version"]),
    ("Account & Access", ["account", "login", "authentication", "password", "subscription", "license", "token", "credential", "access"]),
    ("Billing & Payment", ["billing", "payment", "invoice", "refund", "charge", "dispute", "fee", "transaction", "pricing"]),
    ("Returns & Logistics", ["return", "exchange", "warranty", "delivery", "shipping", "order", "packaging"]),
    ("Product & Documentation", ["product", "feature", "documentation", "guidance", "manual", "specification", "training", "information", "tool", "guideline"]),
    ("Sales & Marketing", ["sales", "marketing", "lead", "campaign", "investment", "finance", "strategy", "promotion", "agency", "portfolio"]),
    ("HR & Organization", ["hr", "human resources", "organization", "employee", "collaboration", "process", "feedback", "team"]),
    ("Incident & Recovery", ["incident", "outage", "emergency", "recovery", "maintenance", "resolution", "escalation", "troubleshooting", "disruption", "notification", "it", "tech support", "technical", "support", "assistance", "service", "issue", "workflow", "reliability"]),
]
DOMAIN_NAMES = [name for name, _ in DOMAIN_RULES] + [GENERAL]
DOMAIN_IDS = {name: i + 1 for i, name in enumerate(DOMAIN_NAMES)}
SHORT_KEYWORDS = {"it", "hr", "api", "vpn", "dns", "lan", "wan", "sla", "crm", "ml", "ai", "ci/cd"}

IMPACT_KW = ("outage", "data loss", "data breach", "security breach", "critical", "widespread", "production", "server room", "all users", "entire", "unable to access", "multiple devices", "kritisch", "ausfall", "sicherheitsvorfall", "datenverlust")
URGENT_KW = ("urgent", "immediately", "as soon as possible", "asap", "critical", "offline", "blocking", "severe", "emergency", "promptly", "dringend", "sofort", "umgehend", "eilt")

TEAMS = [
    ("Network & Infra", ["Network & Connectivity", "Hardware & Devices", "Incident & Recovery"]),
    ("Platform & Cloud", ["Cloud & Platform", "Software & Bugs", "Incident & Recovery"]),
    ("Security", ["Security & Compliance", "Network & Connectivity", "Incident & Recovery"]),
    ("Apps & Endpoints", ["Hardware & Devices", "Software & Bugs", "Product & Documentation", "Account & Access"]),
    ("Business Desk", ["Billing & Payment", "Returns & Logistics", "Sales & Marketing", "HR & Organization", "Account & Access"]),
]

TICKET_COLS = ["ticket_id", "subject", "body", "body_zh", "answer", "language", "tags", "type", "queue", "priority",
               "impact", "urgency", "required_skills", "status", "created_at", "sla_due", "resolved_at",
               "assignee_id", "split", "source_file"]
DOMAIN_COLS = ["domain_id", "name", "aliases", "parent_id"]
ENGINEER_COLS = ["engineer_id", "name", "team", "level", "capacity", "timezone", "active"]
ENGINEER_SKILL_COLS = ["engineer_id", "domain_id", "proficiency"]
SHIFT_COLS = ["shift_id", "engineer_id", "shift_start", "shift_end", "on_call", "timezone"]

PLACEHOLDER_RE = re.compile(r"<[^<>]{1,40}>|\[(?:Your Name|Ihr Name|Name)\]", re.IGNORECASE)


def connect(dbname: str):
    return psycopg2.connect(**PG_CONF, dbname=dbname)


def ensure_database() -> None:
    conn = connect("postgres")
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        if cur.fetchone() is None:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
            print(f"  已创建数据库 {DB_NAME}")
    conn.close()


def run_sql_file(conn, filename: str) -> None:
    text = (SQL_DIR / filename).read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(text)
    conn.commit()


def clean_text(value) -> str:
    if not isinstance(value, str):
        return ""
    s = value.replace("\\n", "\n").replace("\\r", "").replace("\\t", "\t")
    s = PLACEHOLDER_RE.sub("", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def keyword_hit(tag_lower: str, kw: str) -> bool:
    return tag_lower == kw if (len(kw) <= 3 or kw in SHORT_KEYWORDS) else kw in tag_lower


def domain_for(tag: str) -> str:
    t = tag.strip().lower()
    for name, kws in DOMAIN_RULES:
        if any(keyword_hit(t, k) for k in kws):
            return name
    return GENERAL


def build_data():
    df = pd.read_csv(RAW, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    n = len(df)
    rng = np.random.default_rng(SEED)
    tag_cols = [f"tag_{i}" for i in range(1, 9)]

    bodies = df["body"].map(clean_text).tolist()
    answers = df["answer"].map(clean_text).tolist()
    raw_subjects = df["subject"].map(clean_text).tolist()
    subjects = [s if s else b.split("\n")[0][:60] for s, b in zip(raw_subjects, bodies)]

    tag_lists = []
    for row in df[tag_cols].itertuples(index=False, name=None):
        seen, out = set(), []
        for v in row:
            if isinstance(v, str) and v.strip():
                key = v.strip().lower()
                if key not in seen:
                    seen.add(key)
                    out.append(v.strip())
        tag_lists.append(out)

    domain_rows = []
    alias_map = {name: [] for name in DOMAIN_NAMES}
    for tag in sorted({t for tags in tag_lists for t in tags}, key=str.lower):
        alias_map[domain_for(tag)].append(tag)
    for name in DOMAIN_NAMES:
        domain_rows.append((DOMAIN_IDS[name], name, alias_map[name], None))

    required = []
    for tags in tag_lists:
        ids = sorted({DOMAIN_IDS[domain_for(t)] for t in tags})
        required.append(ids or [DOMAIN_IDS[GENERAL]])

    def has_kw(text: str, kws) -> bool:
        low = text[:2000].lower()
        return any(k in low for k in kws)

    impacts, urgencies = [], []
    for body, typ, prio in zip(bodies, df["type"].tolist(), df["priority"].tolist()):
        impacts.append(3 if has_kw(body, IMPACT_KW) else (2 if prio == "high" or typ == "Incident" else 1))
        urgencies.append(3 if has_kw(body, URGENT_KW) else (2 if prio == "high" else 1))

    split_arr = np.empty(n, dtype=object)
    for idx in df.groupby(["queue", "priority", "language"], sort=True).indices.values():
        idx = np.asarray(idx)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * 0.10)))
        n_val = max(1, int(round(len(idx) * 0.10)))
        split_arr[idx[:n_test]] = "test"
        split_arr[idx[n_test:n_test + n_val]] = "val"
        split_arr[idx[n_test + n_val:]] = "train"

    ages = rng.uniform(0, 90 * 24 * 3600, n)
    created = [BASE_TIME - timedelta(seconds=float(s)) for s in ages]
    sla_hours = {"high": 4, "medium": 24, "low": 72}
    sla_due = [c + timedelta(hours=sla_hours[p]) for c, p in zip(created, df["priority"].tolist())]
    statuses = rng.choice(["closed", "open", "pending"], size=n, p=[0.82, 0.12, 0.06]).tolist()
    resolved = [c + timedelta(hours=float(rng.uniform(0.5, 12))) if s == "closed" else None
                for c, s in zip(created, statuses)]

    ticket_rows = list(zip(
        range(1, n + 1), subjects, bodies, [None] * n, answers, df["language"].tolist(), tag_lists,
        df["type"].tolist(), df["queue"].tolist(), df["priority"].tolist(), impacts, urgencies, required,
        statuses, created, sla_due, resolved, [None] * n, split_arr.tolist(), [RAW.name] * n,
    ))

    engineer_rows, skill_rows = [], []
    for i in range(12):
        eid = i + 1
        team, domains = TEAMS[i % len(TEAMS)]
        engineer_rows.append((eid, f"E{eid:02d}", team, int(rng.integers(1, 4)),
                              int(rng.integers(3, 9)), "Asia/Shanghai", True))
        k = int(rng.integers(2, min(4, len(domains)) + 1))
        for d in rng.choice(domains, size=k, replace=False).tolist():
            skill_rows.append((eid, DOMAIN_IDS[d], int(rng.integers(3, 6))))

    shift_rows = []
    sid = 1
    for day in range(7):
        day0 = BASE_TIME + timedelta(days=day)
        for shift_idx, (start_h, end_h) in enumerate([(9, 17), (17, 25)]):
            for j in range(3):
                eid = ((day * 2 + shift_idx) * 3 + j) % 12 + 1
                shift_rows.append((sid, eid, day0 + timedelta(hours=start_h),
                                   day0 + timedelta(hours=end_h), shift_idx == 1, "Asia/Shanghai"))
                sid += 1

    return {
        "skill_domains": (DOMAIN_COLS, domain_rows),
        "engineers": (ENGINEER_COLS, engineer_rows),
        "tickets": (TICKET_COLS, ticket_rows),
        "engineer_skills": (ENGINEER_SKILL_COLS, skill_rows),
        "shifts": (SHIFT_COLS, shift_rows),
    }


def dump_intermediate(data) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_cols = {"skill_domains": ["aliases"], "tickets": ["tags", "required_skills"]}
    for table, (cols, rows) in data.items():
        frame = pd.DataFrame(rows, columns=cols).copy()
        for col in json_cols.get(table, []):
            frame[col] = frame[col].map(lambda v: json.dumps(v, ensure_ascii=False))
        frame.to_csv(OUT / f"{table}.csv", index=False, encoding="utf-8-sig")


def insert_rows(conn, table, cols, rows) -> None:
    statement = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
        sql.Identifier(table), sql.SQL(", ").join(map(sql.Identifier, cols))
    ).as_string(conn)
    rows = [tuple("{}" if isinstance(v, list) and not v else v for v in row) for row in rows]
    with conn.cursor() as cur:
        execute_values(cur, statement, rows, page_size=1000)
    conn.commit()


def verify(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""SELECT 'skill_domains', count(*) FROM skill_domains
                       UNION ALL SELECT 'engineers', count(*) FROM engineers
                       UNION ALL SELECT 'tickets', count(*) FROM tickets
                       UNION ALL SELECT 'engineer_skills', count(*) FROM engineer_skills
                       UNION ALL SELECT 'shifts', count(*) FROM shifts
                       UNION ALL SELECT 'resolution_knowledge', count(*) FROM resolution_knowledge
                       UNION ALL SELECT 'triage_results', count(*) FROM triage_results
                       UNION ALL SELECT 'assignments', count(*) FROM assignments""")
        counts = dict(cur.fetchall())
        cur.execute("SELECT split, count(*) FROM tickets GROUP BY split ORDER BY split")
        splits = dict(cur.fetchall())
        cur.execute("SELECT count(*) FROM tickets WHERE required_skills <> '{}'::integer[]")
        with_skills = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM tickets WHERE subject IS NULL OR subject = ''")
        empty_subject = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM pg_indexes WHERE schemaname = 'public'")
        indexes = cur.fetchone()[0]
    print("  表行数:", counts)
    print("  split 分布:", splits)
    print(f"  已映射技能域工单: {with_skills} / {counts['tickets']}；空标题: {empty_subject}；索引数: {indexes}")


def main() -> None:
    print("[1/6] 检查/创建数据库")
    ensure_database()
    conn = connect(DB_NAME)
    print("[2/6] 建表（8 张表 + vector 扩展）")
    run_sql_file(conn, "01_schema.sql")
    print("[3/6] 清洗 CSV、归并技能域、划分 split、造业务数据")
    data = build_data()
    dump_intermediate(data)
    print("[4/6] 清空并重灌数据")
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE assignments, triage_results, resolution_knowledge, shifts, "
                    "engineer_skills, tickets, engineers, skill_domains RESTART IDENTITY CASCADE")
    conn.commit()
    for table in ("skill_domains", "engineers", "tickets", "engineer_skills", "shifts"):
        cols, rows = data[table]
        insert_rows(conn, table, cols, rows)
        print(f"  {table}: {len(rows)} 行")
    print("[5/6] 建索引")
    run_sql_file(conn, "02_indexes.sql")
    with conn.cursor() as cur:
        cur.execute("ANALYZE")
    conn.commit()
    print("[6/6] 校验")
    verify(conn)
    conn.close()
    print("完成。中间产物在 data/processed/，向量库请用 scripts/embed_knowledge.py 灌入。")


if __name__ == "__main__":
    main()

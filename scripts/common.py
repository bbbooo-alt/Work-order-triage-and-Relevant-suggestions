# -*- coding: utf-8 -*-
"""第一步共用的路径、数据库连接与工具函数。"""
from __future__ import annotations

import datetime
import decimal
import json
import os
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
REPORTS_DIR = ROOT / "reports"
WEB_DIR = ROOT / "web"
EVAL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

DB_NAME = os.environ.get("ITSM_DB", "itsm_triage")
PG_CONF = dict(
    host=os.environ.get("PGHOST", "127.0.0.1"),
    port=int(os.environ.get("PGPORT", "5432")),
    user=os.environ.get("PGUSER", "postgres"),
)
SPLIT_FILE = EVAL_DIR / "split_frozen.csv"
THRESHOLD_FILE = EVAL_DIR / "thresholds.json"
REPORT_FILE = REPORTS_DIR / "baseline_metrics.json"

BASELINE_MODELS = ["majority-v0", "keyword-v0"]
ROUTING_MODEL = "round-robin-v0"


def connect(dbname: str | None = None):
    return psycopg2.connect(**PG_CONF, dbname=dbname or DB_NAME)


def json_default(value):
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    raise TypeError(f"无法序列化: {type(value)}")


def dump_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

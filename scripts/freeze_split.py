# -*- coding: utf-8 -*-
"""冻结 train/val/test 划分：导出只读清单并校验无泄漏。"""
from __future__ import annotations

import csv
from collections import Counter

from common import SPLIT_FILE, connect


def run() -> dict:
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("SELECT ticket_id, split FROM tickets ORDER BY ticket_id")
        rows = cur.fetchall()
        cur.execute("""SELECT count(*) FROM (
                         SELECT body FROM tickets GROUP BY body HAVING count(DISTINCT split) > 1
                       ) x""")
        cross_split_bodies = cur.fetchone()[0]
    conn.close()

    with SPLIT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticket_id", "split"])
        writer.writerows(rows)

    counts = Counter(split for _, split in rows)
    ids = [r[0] for r in rows]
    total = len(rows)
    train_ratio = counts["train"] / total if total else 0
    result = {
        "file": str(SPLIT_FILE),
        "total": total,
        "counts": {k: counts.get(k, 0) for k in ("train", "val", "test")},
        "unique_ticket_ids": len(set(ids)) == len(ids),
        "valid_labels": set(counts) <= {"train", "val", "test"},
        "cross_split_same_body": cross_split_bodies,
        "train_ratio": round(train_ratio, 4),
    }
    result["passed"] = bool(
        result["unique_ticket_ids"]
        and result["valid_labels"]
        and cross_split_bodies == 0
        and 0.75 <= train_ratio <= 0.85
    )
    return result


def check() -> dict:
    """只读校验：冻结清单与数据库是否一致。"""
    if not SPLIT_FILE.exists():
        return {"passed": False, "reason": "冻结清单不存在，请先运行冻结"}
    with SPLIT_FILE.open("r", encoding="utf-8") as f:
        frozen = {int(r["ticket_id"]): r["split"] for r in csv.DictReader(f)}
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("SELECT ticket_id, split FROM tickets")
        current = dict(cur.fetchall())
    conn.close()
    mismatch = sum(1 for k, v in frozen.items() if current.get(k) != v)
    result = {
        "frozen_rows": len(frozen),
        "db_rows": len(current),
        "mismatch_rows": mismatch,
        "missing_in_db": len(set(frozen) - set(current)),
    }
    result["passed"] = (mismatch == 0 and len(frozen) == len(current))
    return result


if __name__ == "__main__":
    print(run())

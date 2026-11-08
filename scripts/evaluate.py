# -*- coding: utf-8 -*-
"""统一评测脚本：分类／优先级指标、路由指标、泄漏与隔离检查。任何后续模型必须复用本脚本。"""
from __future__ import annotations

import datetime
import statistics

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import freeze_split
from common import (BASELINE_MODELS, REPORT_FILE, ROUTING_MODEL, THRESHOLD_FILE,
                    connect, dump_json, load_json)

PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2}
TASKS = [("type", 1, 6), ("queue", 2, 7), ("priority", 3, 8)]


def _classification_metrics(y_true, y_pred, labels):
    macro = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    per = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    result = {
        "macro_f1": round(float(macro), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "per_class_f1": {label: round(float(value), 4) for label, value in zip(labels, per)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": list(labels),
    }
    return result


def run(extra_models=None, report_path=None) -> dict:
    """默认只评第一步的两条基线；传入 extra_models 可加入新模型，report_path 可指定独立报告文件。"""
    model_versions = BASELINE_MODELS + list(extra_models or [])
    conn = connect()
    cur = conn.cursor()
    labels = {}
    for column in ("type", "queue", "priority"):
        cur.execute(f"SELECT DISTINCT {column} FROM tickets ORDER BY 1")
        labels[column] = [r[0] for r in cur.fetchall()]

    cur.execute("""SELECT t.ticket_id, t.type, t.queue, t.priority, t.language,
                          r.model_version, r.pred_type, r.pred_queue, r.pred_priority
                   FROM tickets t JOIN triage_results r USING (ticket_id)
                   WHERE t.split = 'test' AND r.model_version = ANY(%s)""", (model_versions,))
    rows = cur.fetchall()

    models = {}
    for model_version in model_versions:
        subset = [r for r in rows if r[5] == model_version]
        entry = {"rows": len(subset)}
        for task, ti, pi in TASKS:
            y_true = [r[ti] for r in subset]
            y_pred = [r[pi] for r in subset]
            entry[task] = _classification_metrics(y_true, y_pred, labels[task])
            if task == "priority" and y_true:
                diffs = [abs(PRIORITY_ORDER.get(a, 1) - PRIORITY_ORDER.get(b, 1))
                         for a, b in zip(y_true, y_pred)]
                entry[task]["mae"] = round(sum(diffs) / len(diffs), 4)
                entry[task]["severe_error_rate"] = round(sum(d >= 2 for d in diffs) / len(diffs), 4)
        entry["by_language"] = {}
        for language in ("en", "de"):
            part = [r for r in subset if r[4] == language]
            entry["by_language"][language] = {
                task: round(float(f1_score([r[ti] for r in part], [r[pi] for r in part],
                                           average="macro", labels=labels[task], zero_division=0)), 4)
                for task, ti, pi in TASKS
            }
        models[model_version] = entry

    cur.execute("""SELECT count(*),
                          count(*) FILTER (WHERE ea.valid),
                          count(*) FILTER (WHERE ea.finished_at > ea.decided_at +
                              CASE t.priority WHEN 'high' THEN interval '4 hours'
                                             WHEN 'medium' THEN interval '24 hours'
                                             ELSE interval '72 hours' END)
                   FROM eval_assignments ea JOIN tickets t USING (ticket_id)
                   WHERE ea.model_version = %s""", (ROUTING_MODEL,))
    total, valid, timeout = cur.fetchone()
    cur.execute("SELECT engineer_id, count(*) FROM eval_assignments WHERE model_version = %s GROUP BY 1",
                (ROUTING_MODEL,))
    per_engineer = dict(cur.fetchall())
    cur.execute("SELECT engineer_id FROM engineers")
    loads = [per_engineer.get(r[0], 0) for r in cur.fetchall()]

    routing = {
        "model_version": ROUTING_MODEL,
        "tickets": total,
        "valid_rate": round(valid / total, 4) if total else 0,
        "timeout_rate": round(timeout / total, 4) if total else 0,
        "load_std": round(statistics.pstdev(loads), 2) if loads else 0,
        "load_min": min(loads) if loads else 0,
        "load_max": max(loads) if loads else 0,
    }

    cur.execute("SELECT count(*) FROM assignments")
    assignments_rows = cur.fetchone()[0]
    cur.execute("""SELECT count(*) FROM triage_results r
                   WHERE r.model_version = ANY(%s)
                     AND r.ticket_id NOT IN (SELECT ticket_id FROM tickets WHERE split = 'test')""",
                (model_versions,))
    stray_predictions = cur.fetchone()[0]
    cur.execute("SELECT split, count(*) FROM tickets GROUP BY 1")
    split_counts = dict(cur.fetchall())
    conn.close()

    split_check = freeze_split.check()
    thresholds = load_json(THRESHOLD_FILE) if THRESHOLD_FILE.exists() else {}
    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_versions": model_versions,
        "split_counts": split_counts,
        "split_check": split_check,
        "models": models,
        "routing": routing,
        "isolation": {
            "assignments_rows": assignments_rows,
            "stray_predictions_outside_test": stray_predictions,
            "eval_table": "eval_assignments",
        },
        "thresholds": thresholds,
    }
    dump_json(report_path or REPORT_FILE, report)
    return report


if __name__ == "__main__":
    result = run()
    print("报告已写入", REPORT_FILE)
    for name, metrics in result["models"].items():
        print(name, {k: metrics[k]["macro_f1"] for k in ("type", "queue", "priority")})
    print("路由", result["routing"])

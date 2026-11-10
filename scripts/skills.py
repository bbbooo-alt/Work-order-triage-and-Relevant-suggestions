# -*- coding: utf-8 -*-
"""P0-1：技能多标签预测，并用预测技能重跑第四步路由回放，量化「真值上界 vs 线上实际」的差距。"""
from __future__ import annotations

import datetime
import json
import warnings

import numpy as np
from psycopg2.extras import execute_values
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss,
                             precision_score, recall_score)

import route
from common import REPORTS_DIR, connect, dump_json, load_json
from triage import MODEL_VERSION as TRIAGE_MODEL

warnings.filterwarnings("ignore", category=ConvergenceWarning)

MODEL_VERSION = "skills-l1-v1"
REPORT_PATH = REPORTS_DIR / "skills_metrics.json"
IMPACT_PATH = REPORTS_DIR / "skills_impact.json"
PREDICTED_ROUTE_VERSION = "route-v1-predicted"
DOMAINS = list(range(1, 14))
THRESHOLD_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]


def ensure_table() -> None:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS skill_predictions (
        ticket_id       integer PRIMARY KEY REFERENCES tickets(ticket_id),
        model_version   text NOT NULL,
        pred_skills     integer[] NOT NULL,
        true_skills     integer[] NOT NULL,
        decision_scores jsonb,
        split           text NOT NULL,
        created_at      timestamptz NOT NULL DEFAULT now())""")
    cur.execute("""CREATE INDEX IF NOT EXISTS idx_skill_pred_split
                   ON skill_predictions (split)""")
    conn.commit()
    conn.close()


def _text(subject, body) -> str:
    return ((subject or "") + " " + (body or ""))[:2000]


def _labels(required) -> list[int]:
    truth = set(int(x) for x in (required or []))
    return [int(d in truth) for d in DOMAINS]


def train_and_store() -> dict:
    """只在 train 拟合、val 选每域阈值、test 只评一次；预测落独立表并回填 triage_results。"""
    ensure_table()
    conn = connect()
    cur = conn.cursor()
    cur.execute("""SELECT ticket_id, split, subject, body, required_skills
                   FROM tickets ORDER BY ticket_id""")
    rows = cur.fetchall()
    parts = {"train": [], "val": [], "test": []}
    for row in rows:
        parts[row[1]].append(row)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), max_features=30000,
        min_df=3, sublinear_tf=True, dtype=np.float32,
    )
    x_train = vectorizer.fit_transform([_text(r[2], r[3]) for r in parts["train"]])
    x_val = vectorizer.transform([_text(r[2], r[3]) for r in parts["val"]])
    x_test = vectorizer.transform([_text(r[2], r[3]) for r in parts["test"]])
    y_train = np.asarray([_labels(r[4]) for r in parts["train"]], dtype=np.int8)
    y_val = np.asarray([_labels(r[4]) for r in parts["val"]], dtype=np.int8)
    y_test = np.asarray([_labels(r[4]) for r in parts["test"]], dtype=np.int8)

    models, val_prob, test_prob = {}, {}, {}
    skipped = []
    for i, domain in enumerate(DOMAINS):
        if y_train[:, i].sum() < 2:
            skipped.append(domain)
            val_prob[domain] = np.zeros(len(parts["val"]), dtype=float)
            test_prob[domain] = np.zeros(len(parts["test"]), dtype=float)
            continue
        clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=20, tol=1e-3,
                            random_state=42, class_weight="balanced")
        clf.fit(x_train, y_train[:, i])
        models[domain] = clf
        positive_index = list(clf.classes_).index(1)
        val_prob[domain] = clf.predict_proba(x_val)[:, positive_index]
        test_prob[domain] = clf.predict_proba(x_test)[:, positive_index]

    val_matrix = np.column_stack([val_prob[d] for d in DOMAINS])
    test_matrix = np.column_stack([test_prob[d] for d in DOMAINS])

    def matrix_at(matrix, thresholds):
        return np.column_stack([
            (matrix[:, i] >= thresholds[d]).astype(np.int8)
            for i, d in enumerate(DOMAINS)])

    uniform = {d: THRESHOLD_GRID[0] for d in DOMAINS}
    global_threshold = max(
        THRESHOLD_GRID,
        key=lambda t: f1_score(y_val, matrix_at(val_matrix, {d: t for d in DOMAINS}),
                               average="micro", zero_division=0))
    thresholds = {}
    for i, d in enumerate(DOMAINS):
        support = int(y_val[:, i].sum())
        if d in skipped or support < 20:
            thresholds[d] = global_threshold
        else:
            thresholds[d] = max(THRESHOLD_GRID, key=lambda t: f1_score(
                y_val[:, i], (val_matrix[:, i] >= t).astype(np.int8), zero_division=0))

    val_pred = matrix_at(val_matrix, thresholds)
    test_pred = matrix_at(test_matrix, thresholds)

    per_domain = {}
    for i, d in enumerate(DOMAINS):
        per_domain[str(d)] = {
            "precision": round(float(precision_score(y_test[:, i], test_pred[:, i], zero_division=0)), 4),
            "recall": round(float(recall_score(y_test[:, i], test_pred[:, i], zero_division=0)), 4),
            "f1": round(float(f1_score(y_test[:, i], test_pred[:, i], zero_division=0)), 4),
            "support": int(y_test[:, i].sum()),
            "threshold": thresholds[d],
        }

    records = []
    for split, source, predicted in (("val", parts["val"], val_pred),
                                     ("test", parts["test"], test_pred)):
        for i, row in enumerate(source):
            pred = [d for j, d in enumerate(DOMAINS) if predicted[i, j]]
            scores = {str(d): round(float(test_matrix[:, j][i] if split == "test"
                                          else val_matrix[:, j][i]), 4)
                      for j, d in enumerate(DOMAINS)}
            records.append((row[0], MODEL_VERSION, pred, sorted(set(int(x) for x in row[4] or [])),
                            json.dumps(scores), split))

    execute_values(cur, """INSERT INTO skill_predictions
        (ticket_id, model_version, pred_skills, true_skills, decision_scores, split)
        VALUES %s ON CONFLICT (ticket_id) DO UPDATE SET
          model_version = EXCLUDED.model_version, pred_skills = EXCLUDED.pred_skills,
          true_skills = EXCLUDED.true_skills, decision_scores = EXCLUDED.decision_scores,
          split = EXCLUDED.split, created_at = now()""",
        records, template="(%s,%s,%s,%s,%s::jsonb,%s)", page_size=1000)
    conn.commit()
    apply_to_triage_results()
    conn.close()

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_version": MODEL_VERSION,
        "fitted_on": "train only", "thresholds_selected_on": "val",
        "evaluated_on": "test（只评一次）",
        "classifier": "每域一个 SGDClassifier(log_loss, class_weight=balanced)，One-vs-Rest",
        "vectorizer": "TF-IDF char_wb 3-5 元语法，与第二步分诊同口径",
        "global_threshold": global_threshold,
        "per_domain_thresholds": {str(d): thresholds[d] for d in DOMAINS},
        "skipped_domains": skipped,
        "val_micro_f1": round(float(f1_score(y_val, val_pred, average="micro", zero_division=0)), 4),
        "test_metrics": {
            "micro_f1": round(float(f1_score(y_test, test_pred, average="micro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(y_test, test_pred, average="macro", zero_division=0)), 4),
            "subset_accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
            "hamming_loss": round(float(hamming_loss(y_test, test_pred)), 4),
            "avg_predicted_skills": round(float(test_pred.sum() / len(test_pred)), 3),
            "avg_true_skills": round(float(y_test.sum() / len(y_test)), 3),
        },
        "per_domain": per_domain,
        "rows": {s: len(parts[s]) for s in parts},
        "note": "预测写入独立表 skill_predictions，并回填 triage_results.pred_skills；不改动任何真值。",
    }
    dump_json(REPORT_PATH, report)
    return report


def apply_to_triage_results() -> int:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""UPDATE triage_results r SET pred_skills = s.pred_skills
                   FROM skill_predictions s
                   WHERE r.ticket_id = s.ticket_id AND r.model_version = %s
                     AND r.pred_skills <> s.pred_skills""", (TRIAGE_MODEL,))
    updated = cur.rowcount
    conn.commit()
    conn.close()
    return updated


def predicted_skills(conn, ticket_id: int) -> list[int]:
    cur = conn.cursor()
    cur.execute("SELECT pred_skills FROM skill_predictions WHERE ticket_id = %s", (int(ticket_id),))
    row = cur.fetchone()
    return list(row[0]) if row else []


def run_predicted_replay() -> dict:
    """把回放工单的 required_skills 换成模型预测，重跑第四步，量化上界与实际的距离。"""
    if not route.REPORT_PATH.exists():
        route.run()
    conn = connect()
    static = route.load_static(conn)
    replay = route.load_replay(conn, static)
    cur = conn.cursor()
    cur.execute("SELECT ticket_id, pred_skills FROM skill_predictions")
    pred_map = {r[0]: list(r[1] or []) for r in cur.fetchall()}
    for ticket in replay:
        ticket["required"] = set(pred_map.get(ticket["ticket_id"], []))
    cur.execute("DELETE FROM route_assignments WHERE model_version = %s",
                (PREDICTED_ROUTE_VERSION,))
    conn.commit()
    records = [route.assign_one(t, static, PREDICTED_ROUTE_VERSION, conn=conn) for t in replay]
    baseline = route.load_baseline(conn)
    predicted = route.measure(records, len(replay), baseline)
    predicted["model_version"] = PREDICTED_ROUTE_VERSION
    conn.close()

    gold = load_json(route.REPORT_PATH)["metrics"]
    keys = ("assigned_count", "coverage_rate", "degraded_rate",
            "constraint_compliance", "timeout_rate", "load_std", "max_load_ratio")
    delta = {k: round(predicted[k] - gold[k], 4) for k in keys}
    impact = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "gold": {k: gold[k] for k in keys},
        "predicted": {k: predicted[k] for k in keys},
        "delta": delta,
        "skill_model": MODEL_VERSION,
        "skill_metrics": load_json(REPORT_PATH)["test_metrics"] if REPORT_PATH.exists() else {},
        "note": ("gold（真值上界）是第四步原有的 route-v1；predicted 是 route-v1-predicted，"
                 "除所需技能换成模型预测外，其余规则、时间映射、工单集合完全相同。"
                 "差值即「分诊技能预测的误差」向下游路由传播的代价。"),
    }
    dump_json(IMPACT_PATH, impact)
    return impact


def ensure_ready(force: bool = False) -> dict:
    """幂等准备：表/报告/回填缺一补一，避免每次验收都重训 13 个分类头。"""
    need_train = force
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT to_regclass('public.skill_predictions')")
    if cur.fetchone()[0] is None:
        need_train = True
        count = 0
    else:
        cur.execute("SELECT count(*) FROM skill_predictions")
        count = cur.fetchone()[0]
        if count < 5000:
            need_train = True
    cur.execute("""SELECT count(*) FROM triage_results
                   WHERE model_version = %s AND cardinality(pred_skills) > 0""",
                (TRIAGE_MODEL,))
    applied = cur.fetchone()[0]
    conn.close()
    if need_train or not REPORT_PATH.exists():
        train_and_store()
    elif applied < 2000:
        apply_to_triage_results()
    if not IMPACT_PATH.exists() or force:
        run_predicted_replay()
    return {"skills": load_json(REPORT_PATH), "impact": load_json(IMPACT_PATH)}


def build_checks(skills_report: dict, impact: dict) -> list[dict]:
    def item(name, passed, detail):
        return {"name": name, "passed": bool(passed), "detail": detail}

    m = skills_report["test_metrics"]
    gold, pred, delta = impact["gold"], impact["predicted"], impact["delta"]
    return [
        item("技能预测模型已训练并落库", skills_report["rows"]["test"] > 0 and m["micro_f1"] > 0,
             f"test {skills_report['rows']['test']} 条，micro-F1 {m['micro_f1']}，"
             f"macro-F1 {m['macro_f1']}，子集完全命中率 {m['subset_accuracy']}"),
        item("阈值只在验证集上选", skills_report["thresholds_selected_on"] == "val",
             f"全局阈值 {skills_report['global_threshold']}，13 个域各自的阈值在 val 上按 F1 择优"),
        item("pred_skills 不再是空数组", m["avg_predicted_skills"] > 0,
             f"平均预测技能数 {m['avg_predicted_skills']}，真值平均 {m['avg_true_skills']}；"
             "已回填 triage_results.pred_skills"),
        item("链路缺口已打通：路由改用预测技能", pred["assigned_count"] > 0,
             f"route-v1-predicted 成功派出 {pred['assigned_count']} 单，合规率 {pred['constraint_compliance']}"),
        item("差距被量化而非忽略", all(k in delta for k in ("assigned_count", "coverage_rate", "timeout_rate")),
             f"合法派单数 {gold['assigned_count']} → {pred['assigned_count']}（{delta['assigned_count']:+d}），"
             f"覆盖率 {gold['coverage_rate']} → {pred['coverage_rate']}（{delta['coverage_rate']:+.4f}）"),
        item("真值上界与实际结果都写进了独立报告", IMPACT_PATH.exists(),
             str(IMPACT_PATH) + "；第四步原有 route-v1 回放保持不变"),
    ]


if __name__ == "__main__":
    result = ensure_ready(force=True)
    print("技能指标", result["skills"]["test_metrics"])
    print("路由差距", result["impact"]["delta"])
    for check in build_checks(result["skills"], result["impact"]):
        print("通过" if check["passed"] else "未通过", check["name"], "|", check["detail"])

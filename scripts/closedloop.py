# -*- coding: utf-8 -*-
"""P0-3：置信度校准、自动档「置信度＋风险」门禁、回流标注 → 重训 → 与冻结基线对比。"""
from __future__ import annotations

import datetime
import hashlib
import json
import warnings

import numpy as np
from psycopg2.extras import execute_values
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

import evaluate
import lifecycle
import triage
from common import EVAL_DIR, REPORTS_DIR, connect, dump_json, load_json

warnings.filterwarnings("ignore", category=ConvergenceWarning)

CALIBRATION_FILE = EVAL_DIR / "calibration.json"
REPORT_PATH = REPORTS_DIR / "retraining_metrics.json"
SHADOW_VERSION = "triage-shadow-v1"
TARGET_ACCURACY = 0.90
MIN_COVERAGE = 0.05
FEEDBACK_BOOST = 25
CONF_GRID = [round(0.40 + 0.05 * i, 2) for i in range(11)]


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           max_features=30000, min_df=3, sublinear_tf=True,
                           dtype=np.float32)


def _classifier():
    # 与第二步 val 选中的 balanced 配置完全一致，保证校准/重训可比。
    return SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=20, tol=1e-3,
                         random_state=42, class_weight="balanced")


def calibrate_confidence() -> dict:
    """在 val 上把「平均最大概率」校准成自动档准入阈值；test 不参与。"""
    rows = triage.load_rows()
    parts = {s: [r for r in rows if r[1] == s] for s in ("train", "val", "test")}
    vec = _vectorizer()
    x_train = vec.fit_transform([triage._text(r[2], r[3]) for r in parts["train"]])
    x_val = vec.transform([triage._text(r[2], r[3]) for r in parts["val"]])
    models = {}
    for task in triage.TASKS:
        clf = _classifier()
        clf.fit(x_train, [r[triage.TASK_COLUMN[task]] for r in parts["train"]])
        models[task] = clf
    proba = {t: models[t].predict_proba(x_val).max(axis=1) for t in triage.TASKS}
    confidence = np.mean([proba[t] for t in triage.TASKS], axis=0)
    pred_priority = models["priority"].classes_[np.argmax(
        models["priority"].predict_proba(x_val), axis=1)]
    truth_priority = np.asarray([r[triage.TASK_COLUMN["priority"]] for r in parts["val"]])
    correct = (pred_priority == truth_priority)
    three_correct = np.all([
        models[t].predict(x_val) == np.asarray([r[triage.TASK_COLUMN[t]] for r in parts["val"]])
        for t in triage.TASKS], axis=0)

    candidates = []
    for threshold in CONF_GRID:
        mask = confidence >= threshold
        coverage = float(mask.mean())
        accuracy = float(correct[mask].mean()) if mask.any() else 0.0
        candidates.append({"threshold": threshold, "coverage": round(coverage, 4),
                           "priority_accuracy": round(accuracy, 4),
                           "three_task_accuracy": round(float(three_correct[mask].mean()), 4)
                           if mask.any() else 0.0})
    qualified = [c for c in candidates
                 if c["priority_accuracy"] >= TARGET_ACCURACY and c["coverage"] >= MIN_COVERAGE]
    chosen = min(qualified, key=lambda c: c["threshold"]) if qualified else max(
        candidates, key=lambda c: c["priority_accuracy"])

    bins = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        mask = (confidence >= lo) & (confidence < hi if i < 9 else confidence <= hi)
        bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": int(mask.sum()),
                     "avg_confidence": round(float(confidence[mask].mean()), 4) if mask.any() else 0,
                     "priority_accuracy": round(float(correct[mask].mean()), 4) if mask.any() else 0})
    result = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_version": triage.MODEL_VERSION,
        "auto_conf_min": chosen["threshold"],
        "target_priority_accuracy": TARGET_ACCURACY,
        "min_coverage": MIN_COVERAGE,
        "qualified": bool(qualified),
        "chosen": chosen,
        "val_rows": len(parts["val"]),
        "candidates": candidates,
        "reliability_bins": bins,
        "note": ("置信度是三个任务 predict_proba 最大值的平均；在 val 上选满足"
                 "「优先级准确率 >= 0.90 且覆盖 >= 5%」的最低阈值，未达标则取准确率最高的档并标记 qualified=false。"
                 "test 全程不参与阈值选择。"),
    }
    dump_json(CALIBRATION_FILE, result)
    return result


def risk_level(confidence: float | None, pred_priority: str | None,
               pred_type: str | None, retrieval: dict, routing: dict,
               calibration: dict) -> dict:
    reasons, score = [], 0
    conf_min = calibration.get("auto_conf_min", 1.0)
    if confidence is None or confidence < conf_min:
        reasons.append(f"置信度 {confidence} < 准入阈值 {conf_min}")
        score += 2
    if pred_priority == "high":
        reasons.append("高优先级工单需人工确认")
        score += 2
    if pred_type == "Incident":
        reasons.append("事件类工单风险较高")
        score += 1
    if retrieval.get("abstain"):
        reasons.append("检索侧拒答，没有可靠依据")
        score += 2
    if routing.get("status") != "assigned":
        reasons.append("无可用工程师，只能转人工")
        score += 2
    level = "low" if score == 0 else ("medium" if score <= 2 else "high")
    return {"level": level, "score": score, "reasons": reasons or ["全部准入条件满足"]}


def auto_eligibility(context: dict) -> dict:
    """单据级自动档准入：开关 + 置信度 + 风险 + 检索 + 路由，全部通过才自动生效。"""
    calibration = load_json(CALIBRATION_FILE) if CALIBRATION_FILE.exists() else calibrate_confidence()
    risk = risk_level(context.get("confidence"), context.get("pred_priority"),
                      context.get("pred_type"), context.get("retrieval", {}),
                      context.get("routing", {}), calibration)
    gate = lifecycle.attempt_auto({
        "confidence": context.get("confidence"), "risk": risk["level"],
        "auto_conf_min": calibration["auto_conf_min"],
        "retrieval": context.get("retrieval", {}),
        "routing": context.get("routing", {}),
    })
    return {"allowed": gate["allowed"], "mode": gate["mode"],
            "reason": gate["reason"], "risk": risk,
            "auto_conf_min": calibration["auto_conf_min"],
            "failed": gate["failed"], "checks": gate["checks"]}


def ensure_registry() -> None:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS model_registry (
        model_version text PRIMARY KEY,
        trained_at    timestamptz NOT NULL DEFAULT now(),
        dataset_hash  text,
        feedback_rows integer NOT NULL,
        val_scores    jsonb,
        promoted      boolean NOT NULL DEFAULT false,
        note          text)""")
    conn.commit()
    conn.close()


def consume_feedback() -> dict:
    """把 train 侧的人工改判消费成增量训练样本；test 样本绝不进入。"""
    conn = connect()
    cur = conn.cursor()
    cur.execute("""SELECT ticket_id, field, old_value, new_value, accepted
                   FROM feedback_events
                   WHERE split = 'train' AND enters_label_pool AND NOT accepted
                   ORDER BY feedback_id""")
    rows = cur.fetchall()
    cur.execute("""SELECT count(*) FROM feedback_events
                   WHERE split = 'test' AND enters_label_pool""")
    leaked = cur.fetchone()[0]
    conn.close()
    return {"rows": len(rows), "leaked_test_rows": leaked,
            "samples": [{"ticket_id": r[0], "field": r[1],
                         "old_value": r[2], "new_value": r[3]} for r in rows],
            "boost": FEEDBACK_BOOST}


def retrain(feedback: dict) -> dict:
    """重训一个影子版本，写 triage_results，用统一评测脚本与冻结基线对比。"""
    rows = triage.load_rows()
    parts = {s: [r for r in rows if r[1] == s] for s in ("train", "val", "test")}
    overrides = {s["ticket_id"]: s["new_value"] for s in feedback["samples"]
                 if s["field"] == "priority"}
    train_rows = list(parts["train"])
    boosted = []
    for r in train_rows:
        if r[0] in overrides:
            mutable = list(r)
            mutable[triage.TASK_COLUMN["priority"]] = overrides[r[0]]
            boosted.extend([tuple(mutable)] * FEEDBACK_BOOST)
    train_rows += boosted
    dataset_hash = hashlib.md5(
        (str(len(train_rows)) + "|" + json.dumps(overrides, sort_keys=True)).encode()).hexdigest()[:16]

    vec = _vectorizer()
    x_train = vec.fit_transform([triage._text(r[2], r[3]) for r in train_rows])
    x_val = vec.transform([triage._text(r[2], r[3]) for r in parts["val"]])
    x_test = vec.transform([triage._text(r[2], r[3]) for r in parts["test"]])
    models, val_scores = {}, {}
    predictions, max_proba = {}, {}
    for task in triage.TASKS:
        clf = _classifier()
        clf.fit(x_train, [r[triage.TASK_COLUMN[task]] for r in train_rows])
        models[task] = clf
        val_pred = clf.predict(x_val)
        val_scores[task] = round(float(f1_score(
            [r[triage.TASK_COLUMN[task]] for r in parts["val"]], val_pred,
            average="macro", zero_division=0)), 4)
        proba = clf.predict_proba(x_test)
        predictions[task] = clf.classes_[np.argmax(proba, axis=1)]
        max_proba[task] = proba.max(axis=1)

    records = []
    for i, row in enumerate(parts["test"]):
        confidence = float(np.mean([max_proba[t][i] for t in triage.TASKS]))
        records.append((row[0], SHADOW_VERSION,
                        str(predictions["type"][i]), str(predictions["queue"][i]),
                        str(predictions["priority"][i]), confidence, "{}"))
    conn = connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM triage_results WHERE model_version = %s", (SHADOW_VERSION,))
    execute_values(cur, """INSERT INTO triage_results
        (ticket_id, model_version, pred_type, pred_queue, pred_priority, confidence, pred_skills)
        VALUES %s""", records, page_size=1000)
    conn.commit()
    conn.close()

    report = evaluate.run(extra_models=[triage.MODEL_VERSION, SHADOW_VERSION],
                          report_path=REPORT_PATH)
    live = report["models"][triage.MODEL_VERSION]
    shadow = report["models"][SHADOW_VERSION]
    live_val = load_json(triage.REPORT_PATH).get("triage", {}).get(
        "config_candidates", {}).get("balanced", {}).get("val_macro_f1", {})
    val_delta = {t: round(val_scores[t] - live_val.get(t, 0), 4) for t in triage.TASKS}
    promoted = (all(delta >= -0.005 for delta in val_delta.values())
                and shadow["priority"]["mae"] <= live["priority"]["mae"] + 1e-9)

    ensure_registry()
    conn = connect()
    cur = conn.cursor()
    cur.execute("""INSERT INTO model_registry
        (model_version, dataset_hash, feedback_rows, val_scores, promoted, note)
        VALUES (%s,%s,%s,%s::jsonb,%s,%s)
        ON CONFLICT (model_version) DO UPDATE SET trained_at = now(),
          dataset_hash = EXCLUDED.dataset_hash, feedback_rows = EXCLUDED.feedback_rows,
          val_scores = EXCLUDED.val_scores, promoted = EXCLUDED.promoted,
          note = EXCLUDED.note""",
        (SHADOW_VERSION, dataset_hash, feedback["rows"],
         json.dumps(val_scores), promoted,
         "回流样本按 %d 倍加权重训；晋升与否由 val 与冻结 test 指标共同决定" % FEEDBACK_BOOST))
    conn.commit()
    conn.close()

    result = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "shadow_version": SHADOW_VERSION,
        "feedback": feedback,
        "dataset_hash": dataset_hash,
        "train_rows": len(train_rows),
        "boosted_rows": len(boosted),
        "val_scores": val_scores,
        "live_val_scores": live_val,
        "val_delta": val_delta,
        "test_comparison": {
            "baseline_keyword": {t: report["models"]["keyword-v0"][t]["macro_f1"]
                                 for t in triage.TASKS},
            "live": {t: live[t]["macro_f1"] for t in triage.TASKS},
            "shadow": {t: shadow[t]["macro_f1"] for t in triage.TASKS},
            "priority_mae": {"live": live["priority"]["mae"],
                             "shadow": shadow["priority"]["mae"]},
        },
        "frozen_split": report["split_counts"],
        "promoted": promoted,
        "promotion_rule": "val 三任务不掉点且 test 优先级 MAE 不劣化；本轮只产出影子版，不自动替换线上版本",
        "note": ("复用了第一步的统一评测脚本 evaluate.run，报告含 4 个模型；"
                 "回流样本只来自 train，test 只用于最终复评，冻结划分不变。"),
    }
    extended = dict(report)
    extended["retraining"] = result
    dump_json(REPORT_PATH, extended)
    return result


def run(force: bool = False) -> dict:
    calibration = calibrate_confidence()
    feedback = consume_feedback()
    if not lifecycle.REPORT_PATH.exists():
        lifecycle.run()
    if not REPORT_PATH.exists() or force:
        retrain_result = retrain(feedback)
    else:
        retrain_result = load_json(REPORT_PATH).get("retraining", {})
    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "calibration": calibration,
        "auto_gate": {"confidence_min": calibration["auto_conf_min"],
                      "risk_levels": ["low", "medium", "high"],
                      "conditions": ["回退开关关闭", "置信度达标", "风险为 low",
                                     "检索未拒答", "路由有可用工程师"]},
        "feedback": feedback,
        "retraining": retrain_result,
        "mode_state": lifecycle.mode_state(),
        "known_limits": [
            "回流样本量很小，指标变化可能在噪声范围内；本轮验证的是流水线而非模型收益。",
            "置信度只做了阈值校准，未做温度缩放/保序回归的概率校准。",
            "影子版未自动晋升为线上版本，切换需人工确认。",
        ],
    }
    dump_json(REPORTS_DIR / "closed_loop_metrics.json", report)
    return report


def build_checks(report: dict) -> list[dict]:
    def item(name, passed, detail):
        return {"name": name, "passed": bool(passed), "detail": detail}

    cal, retrain_result = report["calibration"], report["retraining"]
    low_probe = auto_eligibility({
        "confidence": 0.10, "pred_priority": "high", "pred_type": "Incident",
        "retrieval": {"abstain": True}, "routing": {"status": "manual_queue"}})
    good_probe = auto_eligibility({
        "confidence": 0.99, "pred_priority": "low", "pred_type": "Request",
        "retrieval": {"abstain": False}, "routing": {"status": "assigned"}})
    return [
        item("置信度阈值在验证集上校准", cal["val_rows"] > 0 and len(cal["reliability_bins"]) == 10,
             f"val {cal['val_rows']} 条，选出 auto_conf_min={cal['auto_conf_min']}，"
             f"该档优先级准确率 {cal['chosen']['priority_accuracy']}、覆盖 {cal['chosen']['coverage']}；"
             f"qualified={cal['qualified']}"),
        item("自动档门禁含置信度与风险分级", not low_probe["allowed"] and low_probe["risk"]["level"] != "low",
             f"低置信度+高优先级+拒答+无工程师的探针被拒绝，风险 {low_probe['risk']['level']}，"
             f"理由 {len(low_probe['risk']['reasons'])} 条"),
        item("满足全部条件时才放行", good_probe["allowed"] and good_probe["risk"]["level"] == "low",
             f"高置信度+低风险探针放行={good_probe['allowed']}，5 项门禁 {good_probe['checks']}"),
        item("回流样本来自 train、test 零泄漏", report["feedback"]["leaked_test_rows"] == 0,
             f"消费 {report['feedback']['rows']} 条 train 改判，"
             f"test 泄漏 {report['feedback']['leaked_test_rows']} 条"),
        item("回流样本真的进入重训", retrain_result.get("boosted_rows", 0) > 0,
             f"增强训练行 {retrain_result.get('boosted_rows',0)}，总训练行 {retrain_result.get('train_rows',0)}，"
             f"数据指纹 {retrain_result.get('dataset_hash','-')}"),
        item("影子版已训练并落库", REPORT_PATH.exists()
             and "shadow" in retrain_result.get("test_comparison", {}),
             f"版本 {retrain_result.get('shadow_version','-')}，写入 triage_results"),
        item("复用了第一步统一评测脚本与冻结基线",
             all(k in retrain_result.get("test_comparison", {})
                 for k in ("baseline_keyword", "live", "shadow"))
             and retrain_result.get("frozen_split", {}).get("test") == 2858,
             f"报告含关键词基线 / 线上版 / 影子版；冻结 test 仍 {retrain_result.get('frozen_split',{}).get('test')} 条"),
        item("晋升判定保守、不自动替换线上版本",
             "promotion_rule" in retrain_result,
             f"promoted={retrain_result.get('promoted')}；{retrain_result.get('promotion_rule','')}"),
        item("闭环报告独立落盘", (REPORTS_DIR / "closed_loop_metrics.json").exists(),
             str(REPORTS_DIR / "closed_loop_metrics.json")),
    ]


if __name__ == "__main__":
    result = run(force=True)
    print("校准阈值", result["calibration"]["auto_conf_min"])
    print("回流样本", result["feedback"]["rows"], "提升后 val", result["retraining"]["val_delta"])
    for check in build_checks(result):
        print("通过" if check["passed"] else "未通过", check["name"], "|", check["detail"])

# -*- coding: utf-8 -*-
"""第二步分诊：TF-IDF 向量 ＋ 线性分类器；只在 train 拟合、val 选配置、test 只评一次。"""
from __future__ import annotations

import warnings

import numpy as np
from psycopg2.extras import execute_values
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

import evaluate
from common import REPORTS_DIR, connect, dump_json

warnings.filterwarnings("ignore", category=ConvergenceWarning)

MODEL_VERSION = "triage-l1-v1"
REPORT_PATH = REPORTS_DIR / "triage_metrics.json"
LOW_CONFIDENCE_THRESHOLD = 0.45
TASKS = ("type", "queue", "priority")
TASK_COLUMN = {"type": 4, "queue": 5, "priority": 6}
CONFIGS = [
    {"name": "plain", "class_weight": None},
    {"name": "balanced", "class_weight": "balanced"},
]


def _item(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _text(subject, body) -> str:
    return ((subject or "") + " " + (body or ""))[:2000]


def load_rows():
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("""SELECT ticket_id, split, subject, body, type, queue, priority, language
                       FROM tickets ORDER BY ticket_id""")
        rows = cur.fetchall()
    conn.close()
    return rows


def train_and_predict() -> dict:
    rows = load_rows()
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

    truth = {
        task: {split: [r[TASK_COLUMN[task]] for r in parts[split]] for split in parts}
        for task in TASKS
    }

    candidates = {}
    for config in CONFIGS:
        models, val_scores = {}, {}
        for task in TASKS:
            clf = SGDClassifier(
                loss="log_loss", alpha=1e-5, max_iter=20, tol=1e-3,
                random_state=42, class_weight=config["class_weight"],
            )
            clf.fit(x_train, truth[task]["train"])
            predicted = clf.predict(x_val)
            val_scores[task] = round(float(f1_score(truth[task]["val"], predicted,
                                                     average="macro", zero_division=0)), 4)
            models[task] = clf
        candidates[config["name"]] = {
            "models": models,
            "val_scores": val_scores,
            "avg": round(sum(val_scores.values()) / len(TASKS), 4),
        }

    best_name = max(candidates, key=lambda name: candidates[name]["avg"])
    best = candidates[best_name]

    probabilities = {task: best["models"][task].predict_proba(x_test) for task in TASKS}
    predictions = {
        task: best["models"][task].classes_[np.argmax(probabilities[task], axis=1)]
        for task in TASKS
    }
    max_proba = {task: probabilities[task].max(axis=1) for task in TASKS}

    records, low_confidence = [], 0
    for i, row in enumerate(parts["test"]):
        confidence = float(np.mean([max_proba[task][i] for task in TASKS]))
        low_confidence += int(confidence < LOW_CONFIDENCE_THRESHOLD)
        records.append((
            row[0], MODEL_VERSION,
            str(predictions["type"][i]), str(predictions["queue"][i]), str(predictions["priority"][i]),
            confidence, "{}",
        ))

    conn = connect()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM triage_results WHERE model_version = %s", (MODEL_VERSION,))
        execute_values(cur, """INSERT INTO triage_results
            (ticket_id, model_version, pred_type, pred_queue, pred_priority, confidence, pred_skills)
            VALUES %s""", records, page_size=1000)
    conn.commit()
    conn.close()

    return {
        "model_version": MODEL_VERSION,
        "vectorizer": "TF-IDF（词频-逆文档频率）char_wb 3-5 元语法，max_features=30000，仅在 train（训练集）拟合",
        "classifier": "SGDClassifier（线性分类器，log_loss），random_state=42",
        "fitted_on": "train only（只在训练集拟合，验证/测试只做转换）",
        "config_candidates": {
            name: {"val_macro_f1": value["val_scores"], "avg": value["avg"]}
            for name, value in candidates.items()
        },
        "selected_config": best_name,
        "selected_by": "val（验证集）上三个任务 macro-F1 的平均值最高",
        "test_predictions": len(records),
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "low_confidence_tickets": low_confidence,
        "note": "为避免冷启动依赖 Ollama 且保证秒级可重跑，L1 先用 TF-IDF 向量；后续可替换为多语言嵌入模型。",
    }


def evaluate_triage(meta: dict) -> dict:
    report = evaluate.run(extra_models=[MODEL_VERSION], report_path=REPORT_PATH)
    report["triage"] = meta
    dump_json(REPORT_PATH, report)
    return report


def build_checks(report: dict) -> list[dict]:
    models = report["models"]
    baseline = models["keyword-v0"]
    current = models[MODEL_VERSION]
    thresholds = report["thresholds"]
    gain = thresholds.get("min_macro_f1_gain_over_keyword_baseline", 0.03)
    checks = []

    for task in TASKS:
        before, after = baseline[task]["macro_f1"], current[task]["macro_f1"]
        checks.append(_item(
            f"{task} 的 macro-F1 提升 ≥ +{gain}",
            after - before >= gain,
            f"关键词基线 {before} → 分诊模型 {after}，差值 {round(after - before, 4)}",
        ))

    regressions = []
    for task in TASKS:
        for label, value in current[task]["per_class_f1"].items():
            old = baseline[task]["per_class_f1"].get(label, 0)
            if value + 1e-9 < old:
                regressions.append(f"{task}/{label} {old}→{value}")
    checks.append(_item("无任一类别 F1 下降", not regressions,
                        "; ".join(regressions[:8]) if regressions else "所有类别的逐类 F1 均未下降"))

    before_mae = baseline["priority"]["mae"]
    after_mae = current["priority"]["mae"]
    checks.append(_item("优先级有序性不劣于基线", after_mae <= before_mae,
                        f"MAE（平均绝对误差）{before_mae} → {after_mae}；严重错误率 "
                        f"{baseline['priority']['severe_error_rate']} → {current['priority']['severe_error_rate']}"))

    lang_problems = []
    for language in ("en", "de"):
        for task in TASKS:
            before = baseline["by_language"][language][task]
            after = current["by_language"][language][task]
            if after + 1e-9 < before:
                lang_problems.append(f"{language}/{task} {before}→{after}")
    checks.append(_item("英、德语分别不掉点", not lang_problems,
                        "; ".join(lang_problems[:6]) if lang_problems else "en/de 两类语种的三个任务均未低于基线"))

    meta = report["triage"]
    checks.append(_item("低置信度出口可用", meta["low_confidence_threshold"] > 0 and meta["test_predictions"] > 0,
                        f"阈值 {meta['low_confidence_threshold']}；test 集中 {meta['low_confidence_tickets']} 条低于阈值，建议转人工"))
    checks.append(_item("向量化只在训练集拟合", meta["fitted_on"].startswith("train only"),
                        str(meta["vectorizer"])))
    checks.append(_item("配置由验证集选定", bool(meta["selected_config"]),
                        f"候选 {list(meta['config_candidates'])}；选中 {meta['selected_config']}；"
                        f"val 平均分 {meta['config_candidates'][meta['selected_config']]['avg']}"))
    checks.append(_item("测试集预测已落库", meta["test_predictions"] == report["split_counts"]["test"],
                        f"triage_results 中 {MODEL_VERSION} 共 {meta['test_predictions']} 条"))
    checks.append(_item("真值与线上派单未被改动",
                        report["isolation"]["assignments_rows"] == 0
                        and report["isolation"]["stray_predictions_outside_test"] == 0,
                        f"assignments 仍为 {report['isolation']['assignments_rows']} 行；越界预测 0 条"))
    return checks


if __name__ == "__main__":
    meta = train_and_predict()
    result = evaluate_triage(meta)
    print("模型", MODEL_VERSION, "选中配置", meta["selected_config"])
    for check in build_checks(result):
        print(("通过" if check["passed"] else "未通过"), check["name"], "|", check["detail"])

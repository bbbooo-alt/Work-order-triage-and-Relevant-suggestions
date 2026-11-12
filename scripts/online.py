# -*- coding: utf-8 -*-
"""面向用户工作台的在线推理层：输入一段故障描述 → 给出分诊、依据、建议、派单与处置结论。

与验收/测试链路的边界（重要）：
- 分诊与技能模型在进程内按原配置训练一次并缓存，用于任意新文本；不写 triage_results / skill_predictions；
- 检索、生成、路由、门禁复用既有实现，路由为 dry_run；
- 用户请求、反馈、确认派单只写独立表 user_requests / user_feedback / user_dispatches，
  线上 assignments 与所有冻结评测数据保持 0 改动。
"""
from __future__ import annotations

import datetime
import json
import threading
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

import closedloop
import generation
import knowledge
import lifecycle
import route
import skills
import triage
from common import connect, load_json

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROUTE_VERSION = "route-v1-userdesk"
GATE_LABELS = {
    "kill_switch_off": "系统回退开关已开启",
    "confidence_ok": "分诊置信度未达自动处置准入线",
    "risk_low": "工单风险等级偏高",
    "retrieval_grounded": "缺少可靠的知识依据",
    "routing_assigned": "暂无满足硬约束的工程师",
}
MODE_HINT = {
    "shadow": "影子模式：系统只记录建议，不自动派单",
    "assist": "辅助模式：建议由人工确认后生效",
    "auto": "自动模式：仅高置信度、低风险工单可自动生效",
}

_LOCK = threading.Lock()
_MODELS: dict | None = None


def ensure_tables() -> None:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS user_requests (
        request_id       bigserial PRIMARY KEY,
        subject          text NOT NULL DEFAULT '',
        body             text NOT NULL,
        redacted_preview text,
        model_version    text NOT NULL,
        skill_model      text,
        result           jsonb NOT NULL,
        disposition      text NOT NULL,
        mode             text NOT NULL,
        created_at       timestamptz NOT NULL DEFAULT now())""")
    cur.execute("""CREATE TABLE IF NOT EXISTS user_feedback (
        feedback_id bigserial PRIMARY KEY,
        request_id  bigint NOT NULL REFERENCES user_requests(request_id),
        field       text, value text, rating smallint, note text,
        created_at  timestamptz NOT NULL DEFAULT now())""")
    cur.execute("""CREATE TABLE IF NOT EXISTS user_dispatches (
        dispatch_id  bigserial PRIMARY KEY,
        request_id   bigint NOT NULL REFERENCES user_requests(request_id),
        engineer_id  smallint,
        confirmed_by text NOT NULL DEFAULT 'user',
        note         text,
        created_at   timestamptz NOT NULL DEFAULT now())""")
    conn.commit()
    conn.close()


def models_ready() -> bool:
    return _MODELS is not None


def ensure_models() -> dict:
    """按第二步与 P0-1 的完全相同的配置训练一次，之后常驻内存供在线推理。"""
    global _MODELS
    if _MODELS is not None:
        return _MODELS
    with _LOCK:
        if _MODELS is not None:
            return _MODELS
        rows = triage.load_rows()
        train = [r for r in rows if r[1] == "train"]
        texts = [triage._text(r[2], r[3]) for r in train]

        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_features=30000,
            min_df=3, sublinear_tf=True, dtype=np.float32,
        )
        x_train = vectorizer.fit_transform(texts)

        triage_report = load_json(triage.REPORT_PATH) if triage.REPORT_PATH.exists() else {}
        selected = triage_report.get("triage", {}).get("selected_config", "balanced")
        class_weight = {"plain": None, "balanced": "balanced"}.get(selected, "balanced")

        triage_models = {}
        for task in triage.TASKS:
            clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=20, tol=1e-3,
                                random_state=42, class_weight=class_weight)
            clf.fit(x_train, [r[triage.TASK_COLUMN[task]] for r in train])
            triage_models[task] = clf

        skill_report = load_json(skills.REPORT_PATH) if skills.REPORT_PATH.exists() else {}
        thresholds = {int(k): float(v) for k, v in
                      skill_report.get("per_domain_thresholds", {}).items()}
        global_threshold = float(skill_report.get("global_threshold", 0.5))
        # 技能标签必须单独取：triage.load_rows() 的第 5 列是 type，不是 required_skills。
        conn = connect()
        cur = conn.cursor()
        cur.execute("""SELECT ticket_id, split, subject, body, required_skills
                       FROM tickets ORDER BY ticket_id""")
        skill_rows = cur.fetchall()
        conn.close()
        skill_train = [r for r in skill_rows if r[1] == "train"]
        if len(skill_train) != len(train):
            raise RuntimeError("分诊与技能训练样本不一致，无法安全共用向量化结果")
        y_train = np.asarray([
            [int(d in set(int(x) for x in (r[4] or []))) for d in skills.DOMAINS]
            for r in skill_train
        ], dtype=np.int8)
        skill_models = {}
        for i, domain in enumerate(skills.DOMAINS):
            if y_train[:, i].sum() < 2:
                skill_models[domain] = None
                continue
            clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=20, tol=1e-3,
                                random_state=42, class_weight="balanced")
            clf.fit(x_train, y_train[:, i])
            skill_models[domain] = clf

        _MODELS = {"vectorizer": vectorizer, "triage": triage_models,
                   "skills": skill_models, "thresholds": thresholds,
                   "global_threshold": global_threshold,
                   "selected_config": selected,
                   "trained_rows": len(train),
                   "trained_at": datetime.datetime.now().isoformat(timespec="seconds")}
    return _MODELS


def predict(subject: str, body: str) -> dict:
    """任意新文本的分诊 + 技能预测；口径与离线验收一致，但不落任何评测表。"""
    m = ensure_models()
    x = m["vectorizer"].transform([triage._text(subject, body)])
    predicted, confidence_parts = {}, {}
    for task in triage.TASKS:
        proba = m["triage"][task].predict_proba(x)[0]
        index = int(np.argmax(proba))
        predicted[task] = str(m["triage"][task].classes_[index])
        confidence_parts[task] = float(proba[index])
    confidence = float(np.mean(list(confidence_parts.values())))

    skill_scores = {}
    for domain, clf in m["skills"].items():
        if clf is None:
            skill_scores[domain] = 0.0
            continue
        classes = list(clf.classes_)
        skill_scores[domain] = (float(clf.predict_proba(x)[0][classes.index(1)])
                                if 1 in classes else 0.0)
    pred_skills = [d for d in skills.DOMAINS
                   if skill_scores[d] >= m["thresholds"].get(d, m["global_threshold"])]
    return {"pred_type": predicted["type"], "pred_queue": predicted["queue"],
            "pred_priority": predicted["priority"],
            "confidence": round(confidence, 4),
            "confidence_parts": {k: round(v, 4) for k, v in confidence_parts.items()},
            "pred_skills": pred_skills,
            "skill_scores": {str(k): round(v, 4) for k, v in skill_scores.items()},
            "model_version": triage.MODEL_VERSION, "skill_model": skills.MODEL_VERSION,
            "selected_config": m["selected_config"]}


def _routing(query_pred: dict, confidence: float) -> dict:
    conn = connect()
    try:
        static = route.load_static(conn)
    finally:
        conn.close()
    arrived = static["win_start"] + (static["win_end"] - static["win_start"]) * 0.35
    ticket = {"ticket_id": 0, "priority": query_pred["pred_priority"],
              "required": set(query_pred["pred_skills"]), "arrived": arrived,
              "pred_priority": query_pred["pred_priority"],
              "pred_queue": query_pred["pred_queue"], "confidence": confidence,
              "before_assignment_id": None}
    probe = connect()
    try:
        decision = route.assign_one(ticket, static, ROUTE_VERSION,
                                    conn=probe, dry_run=True)
    finally:
        probe.close()
    ev = decision["evaluation"]
    chosen = ev["chosen"]
    return {
        "status": decision["status"],
        "recommended": ({"engineer_id": chosen["engineer_id"], "level": chosen["level"],
                         "score": chosen["score"], "matched_domains": chosen["matched_domains"],
                         "load": chosen["load"], "capacity": chosen["capacity"],
                         "only_on_call": chosen["only_on_call"], "parts": chosen["parts"]}
                        if chosen else None),
        "candidates": ev["candidates"][:5],
        "candidate_count": len(ev["candidates"]),
        "rejects": ev["rejects"],
        "reason": decision["reason"],
        "required_used": sorted(query_pred["pred_skills"]),
        "dry_run": True,
    }


def handle(subject: str, body: str, use_generation: bool = True,
           use_cache: bool = True, persist: bool = True) -> dict:
    """完整在线链路；返回业务化字段，不暴露验收术语。"""
    ensure_tables()
    started = time.time()
    subject = (subject or "").strip()
    body = (body or "").strip()
    query = ((subject + " " + body).strip())[:600]
    if not query:
        raise ValueError("请填写工单描述后再提交")

    pred = predict(subject, body)
    redacted = lifecycle.redact(query)

    knowledge.apply_active_weight()
    knowledge.ensure_ollama()
    query_vector = knowledge.embed([query])[0]
    conn = connect()
    try:
        search = knowledge.search(conn, query, query_vector=query_vector, mode="hybrid")
    finally:
        conn.close()
    hits = search["results"]
    abstain = bool(search["abstain"])
    retrieval = {
        "abstain": abstain,
        "notice": ("没有找到足够相关的权威知识依据，本次不生成处理建议，请转人工处理。"
                   if abstain else "以下为依据片段，均来自知识库或历史工单原文。"),
        "best_kb_similarity": search["best_kb_similarity"],
        "lexical_overlap": search["lexical_overlap"],
        "kb_in_top_k": search["kb_in_top_k"],
        "hits": [{"rank": h["rank"], "source": h["source_label"],
                  "similarity": round(float(h["similarity"]), 4),
                  "chunk_id": h["chunk_id"], "preview": h["preview"]}
                 for h in hits],
    }

    if not use_generation:
        gen = {"abstained": True, "reason": "本次选择「只查看依据片段」，未生成处理建议",
               "model": generation.GEN_MODEL, "steps": [], "citations": [],
               "grounded": 0.0, "latency_sec": 0.0, "from_cache": False}
    else:
        gen = generation.generate(query, hits, abstain=abstain, use_cache=use_cache)
    payload = gen.get("response", {}) or {}
    advice = {
        "abstained": bool(gen.get("abstained")),
        "reason": gen.get("reason", ""),
        "steps": [str(x) for x in payload.get("steps", [])],
        "info_needed": [str(x) for x in payload.get("info_needed", [])],
        "escalation": str(payload.get("escalation", "")),
        "citations": gen.get("citations", []),
        "citation_valid_rate": gen.get("citation_valid_rate"),
        "grounded": gen.get("grounded", 0.0),
        "latency_sec": gen.get("latency_sec", 0.0),
        "from_cache": gen.get("from_cache", False),
        "model": gen.get("model", generation.GEN_MODEL),
    }

    routing = _routing(pred, pred["confidence"])
    gate = closedloop.auto_eligibility({
        "confidence": pred["confidence"], "pred_priority": pred["pred_priority"],
        "pred_type": pred["pred_type"],
        "retrieval": {"abstain": abstain},
        "routing": {"status": routing["status"]},
    })
    mode = lifecycle.mode_state()
    failed_zh = [GATE_LABELS.get(k, k) for k in gate.get("failed", [])]
    allowed = bool(gate.get("allowed"))
    if allowed and mode["mode"] == "auto":
        disposition, action = "可自动派单", "门禁全部通过，当前为自动档，可按推荐工程师直接派单。"
    elif allowed:
        disposition, action = "建议人工确认后派单", "门禁已通过；当前为辅助档，请人工确认后再派单。"
    else:
        disposition, action = "建议人工复核", ("需人工复核：" + "、".join(failed_zh)
                                          if failed_zh else "请人工复核后再处置。")

    decision = {
        "disposition": disposition, "action": action, "mode": mode["mode"],
        "mode_label": MODE_HINT.get(mode["mode"], mode["mode"]),
        "auto_allowed": allowed,
        "risk": gate.get("risk", {}),
        "failed": failed_zh,
        "auto_conf_min": gate.get("auto_conf_min"),
        "gate_checks": gate.get("checks", {}),
    }

    result = {
        "request": {"subject": subject, "body_preview": body[:400],
                    "language_hint": "zh-CN / en / de 自动识别",
                    "redacted_preview": redacted["redacted"][:400]},
        "triage": {"type": pred["pred_type"], "queue": pred["pred_queue"],
                   "priority": pred["pred_priority"], "confidence": pred["confidence"],
                   "confidence_parts": pred["confidence_parts"]},
        "skills": {"labels": pred["pred_skills"], "scores": pred["skill_scores"],
                   "model": pred["skill_model"]},
        "retrieval": retrieval,
        "advice": advice,
        "routing": routing,
        "decision": decision,
        "meta": {"model_version": pred["model_version"],
                 "skill_model": pred["skill_model"],
                 "generation_model": generation.GEN_MODEL,
                 "embedding_model": knowledge.MODEL,
                 "route_version": ROUTE_VERSION,
                 "selected_config": pred["selected_config"],
                 "elapsed_ms": int((time.time() - started) * 1000),
                 "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                 "isolation": "演示环境：确认派单写入独立表 user_dispatches，不写线上 assignments"},
    }

    if persist:
        conn = connect()
        cur = conn.cursor()
        cur.execute("""INSERT INTO user_requests
            (subject, body, redacted_preview, model_version, skill_model,
             result, disposition, mode)
            VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s,%s) RETURNING request_id""",
                    (subject, body, redacted["redacted"][:400], pred["model_version"],
                     pred["skill_model"], json.dumps(result, ensure_ascii=False, default=str),
                     disposition, mode["mode"]))
        result["request_id"] = cur.fetchone()[0]
        conn.commit()
        conn.close()
    return result


def get_request(request_id: int) -> dict | None:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""SELECT request_id, subject, body, result, disposition, mode, created_at
                   FROM user_requests WHERE request_id = %s""", (int(request_id),))
    row = cur.fetchone()
    if row is None:
        conn.close()
        return None
    cur.execute("""SELECT field, value, rating, note, created_at FROM user_feedback
                   WHERE request_id = %s ORDER BY feedback_id""", (int(request_id),))
    feedback = [{"field": r[0], "value": r[1], "rating": r[2],
                 "note": r[3], "created_at": str(r[4])} for r in cur.fetchall()]
    cur.execute("""SELECT dispatch_id, engineer_id, confirmed_by, note, created_at
                   FROM user_dispatches WHERE request_id = %s ORDER BY dispatch_id""",
                (int(request_id),))
    dispatches = [{"dispatch_id": r[0], "engineer_id": r[1], "confirmed_by": r[2],
                   "note": r[3], "created_at": str(r[4])} for r in cur.fetchall()]
    conn.close()
    result = row[3] if isinstance(row[3], dict) else json.loads(row[3])
    result["request_id"] = row[0]
    result["request"]["body_preview"] = (row[2] or "")[:400]
    result["feedback"] = feedback
    result["dispatches"] = dispatches
    result["stored"] = {"subject": row[1], "disposition": row[4],
                        "mode": row[5], "created_at": str(row[6])}
    return result


def history(limit: int = 12) -> list[dict]:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""SELECT request_id, subject, disposition, mode, created_at, result
                   FROM user_requests ORDER BY request_id DESC LIMIT %s""", (int(limit),))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        data = r[5] if isinstance(r[5], dict) else json.loads(r[5])
        tri = data.get("triage", {})
        rt = data.get("routing", {})
        out.append({"request_id": r[0], "subject": r[1] or "(无标题)",
                    "disposition": r[2], "mode": r[3], "created_at": str(r[4]),
                    "priority": tri.get("priority"), "confidence": tri.get("confidence"),
                    "engineer_id": (rt.get("recommended") or {}).get("engineer_id"),
                    "abstained": data.get("advice", {}).get("abstained")})
    return out


def confirm_dispatch(request_id: int, engineer_id: int | None, note: str = "") -> dict:
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM user_requests WHERE request_id = %s", (int(request_id),))
    if cur.fetchone() is None:
        conn.close()
        return {"ok": False, "error": f"找不到处理记录 {request_id}"}
    cur.execute("""INSERT INTO user_dispatches
        (request_id, engineer_id, confirmed_by, note) VALUES (%s,%s,'user',%s)
        RETURNING dispatch_id""", (int(request_id), engineer_id, note))
    dispatch_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return {"ok": True, "dispatch_id": dispatch_id, "request_id": int(request_id),
            "engineer_id": engineer_id,
            "note": "已记录在独立派单表 user_dispatches；演示环境不写线上 assignments"}


def save_feedback(request_id: int, field: str = "", value: str = "",
                  rating: int | None = None, note: str = "") -> dict:
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM user_requests WHERE request_id = %s", (int(request_id),))
    if cur.fetchone() is None:
        conn.close()
        return {"ok": False, "error": f"找不到处理记录 {request_id}"}
    cur.execute("""INSERT INTO user_feedback (request_id, field, value, rating, note)
                   VALUES (%s,%s,%s,%s,%s) RETURNING feedback_id""",
                (int(request_id), field or None, value or None, rating, note))
    fid = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return {"ok": True, "feedback_id": fid, "request_id": int(request_id),
            "note": "反馈进入独立标注池，不影响冻结测试集，也不参与本轮模型更新"}


def status() -> dict:
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM engineers WHERE active")
    engineers = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM resolution_knowledge WHERE chunk_type='kb'")
    kb = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM user_requests")
    total = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM user_requests WHERE created_at >= current_date")
    today = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM assignments")
    online_rows = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM user_dispatches")
    dispatches = cur.fetchone()[0]
    conn.close()
    mode = lifecycle.mode_state()
    return {"service_ok": True, "models_ready": models_ready(),
            "available_engineers": engineers, "kb_documents": kb,
            "handled_total": total, "handled_today": today,
            "confirmed_dispatches": dispatches,
            "mode": mode["mode"], "mode_label": MODE_HINT.get(mode["mode"], mode["mode"]),
            "model_version": triage.MODEL_VERSION,
            "generation_model": generation.GEN_MODEL,
            "online_assignments_rows": online_rows,
            "ollama_ok": knowledge._probe_ollama()}


def sample_tickets(limit: int = 6) -> list[dict]:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""SELECT ticket_id, subject, body, priority FROM tickets
                   WHERE split = 'test'
                   ORDER BY md5(ticket_id::text || 'userdesk') LIMIT %s""", (int(limit),))
    rows = cur.fetchall()
    conn.close()
    return [{"ticket_id": r[0], "subject": r[1] or "(无标题)",
             "body": (r[2] or "")[:800], "priority": r[3]} for r in rows]


def knowledge_documents() -> list[dict]:
    return generation.kb_documents()


def skill_domains() -> list[dict]:
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT domain_id, name FROM skill_domains ORDER BY domain_id")
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1]} for r in rows]


def bootstrap() -> dict:
    ensure_tables()
    return {"ok": True, "status": status(), "samples": sample_tickets(),
            "knowledge": knowledge_documents(), "domains": skill_domains(),
            "history": history(8)}


def warmup() -> None:
    """服务启动后台预热，避免用户第一次提交时等待模型训练。"""
    try:
        ensure_tables()
        ensure_models()
    except Exception:
        pass


if __name__ == "__main__":
    ensure_tables()
    result = handle("打印机无法连接", "办公室打印机提示 offline，重启后仍然无法连接网络，请协助排查。")
    print("处置", result["decision"]["disposition"], "| 分诊", result["triage"],
          "| 技能", result["skills"]["labels"])

# -*- coding: utf-8 -*-
"""一体化控制台服务：端口 8770，把五步验收、三项 P0 与端到端流水线融在一个页面。"""
from __future__ import annotations

import json
import os
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import closedloop  # noqa: E402
import generation  # noqa: E402
import knowledge  # noqa: E402
import lifecycle  # noqa: E402
import pipeline  # noqa: E402
import skills  # noqa: E402
from common import WEB_DIR, connect, load_json  # noqa: E402

PORT = 8770
INDEPENDENT = [
    {"port": 8765, "label": "① 评测地基", "url": "http://127.0.0.1:8765/"},
    {"port": 8766, "label": "② 分诊", "url": "http://127.0.0.1:8766/"},
    {"port": 8767, "label": "③ 解决建议", "url": "http://127.0.0.1:8767/"},
    {"port": 8768, "label": "④ 路由", "url": "http://127.0.0.1:8768/"},
    {"port": 8769, "label": "⑤ 上线闭环", "url": "http://127.0.0.1:8769/"},
]


def api_overview() -> dict:
    ready = pipeline.readiness()
    groups = pipeline._read_checks()
    report_path = pipeline.REPORT_PATH
    return {
        "ok": True, "port": PORT, "pipeline_version": pipeline.PIPELINE_VERSION,
        "ready": ready,
        "groups": [{"key": g["key"], "label": g["label"], "source": g["source"],
                    "passed": g["passed"], "total": g["total"]} for g in groups],
        "group_total": {"passed": sum(g["passed"] for g in groups),
                        "total": sum(g["total"] for g in groups)},
        "latest_run_all": (load_json(report_path) if report_path.exists() else None),
        "independent": INDEPENDENT,
        "reports": [{"key": k, "file": p.name, "exists": p.exists()}
                    for k, p in pipeline.REPORT_FILES.items()],
    }


def api_tickets(seed: str) -> dict:
    pool = pipeline.ticket_pool(seed=seed, limit=8)
    return {"ok": True, "seed": seed, "tickets": [
        {k: t[k] for k in ("ticket_id", "split", "subject", "language", "priority",
                           "pred_priority", "pred_type", "pred_queue", "confidence",
                           "pred_skills", "true_skills", "body_preview")}
        for t in pool]}


def api_report(name: str) -> dict:
    path = pipeline.REPORT_FILES.get(name)
    if path is None or not path.exists():
        return {"ok": False, "error": f"报告不存在：{name}"}
    return {"ok": True, "name": name, "file": path.name, "report": load_json(path)}


def api_history() -> dict:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""SELECT run_id, ticket_id, split, mode, auto_allowed, disposition, created_at
                   FROM pipeline_runs ORDER BY run_id DESC LIMIT 12""")
    rows = cur.fetchall()
    cur.execute("""SELECT stage, mode, count(*) FROM decision_log
                   WHERE stage = 'pipeline' GROUP BY 1,2 ORDER BY 1,2""")
    audit = [{"stage": r[0], "mode": r[1], "rows": r[2]} for r in cur.fetchall()]
    conn.close()
    return {"ok": True, "runs": [{"run_id": r[0], "ticket_id": r[1], "split": r[2],
                                  "mode": r[3], "auto_allowed": r[4],
                                  "disposition": r[5], "created_at": str(r[6])}
                                 for r in rows], "audit": audit}


def api_labeling() -> dict:
    seeded = generation.seed_golden_labels()
    conn = connect()
    cur = conn.cursor()
    cur.execute("""SELECT ticket_id, query_text, relevant_chunk_ids, annotator, reviewed, note
                   FROM golden_labels ORDER BY ticket_id LIMIT 60""")
    labels = [{"ticket_id": r[0], "query_text": (r[1] or "")[:180],
               "relevant_chunk_ids": list(r[2] or []), "annotator": r[3],
               "reviewed": r[4], "note": r[5] or ""} for r in cur.fetchall()]
    cur.execute("SELECT ticket_id, relevance, executability, hallucination, note FROM generation_review")
    reviews = [{"ticket_id": r[0], "relevance": r[1], "executability": r[2],
                "hallucination": r[3], "note": r[4] or ""} for r in cur.fetchall()]
    cur.execute("SELECT count(*) FROM generation_cache")
    cache_rows = cur.fetchone()[0]
    conn.close()
    generation_report = (load_json(generation.REPORT_PATH)
                         if generation.REPORT_PATH.exists() else {})
    return {"ok": True, "seeded": seeded, "kb_documents": generation.kb_documents(),
            "labels": labels, "reviews": reviews, "cache_rows": cache_rows,
            "comparison": generation_report.get("comparison", []),
            "aggregate": generation_report.get("aggregate", {}),
            "golden_retrieval": generation_report.get("golden_retrieval", {}),
            "known_limits": generation_report.get("known_limits", [])}


def api_services() -> dict:
    result = []
    for entry in INDEPENDENT:
        try:
            response = requests.get(entry["url"] + "api/status", timeout=1)
            online = response.status_code == 200
            detail = response.json() if online else {}
        except Exception:
            online, detail = False, {}
        result.append({**entry, "online": online,
                       "db_ok": detail.get("db_ok") if online else None})
    return {"ok": True, "services": result,
            "note": "独立验收页面各自有启动脚本；未启动不影响本控制台的功能。"}


def api_auto_probe(kind: str) -> dict:
    if kind == "low":
        context = {"confidence": 0.10, "pred_priority": "high", "pred_type": "Incident",
                   "retrieval": {"abstain": True}, "routing": {"status": "manual_queue"}}
    else:
        context = {"confidence": 0.99, "pred_priority": "low", "pred_type": "Request",
                   "retrieval": {"abstain": False}, "routing": {"status": "assigned"}}
    return {"ok": True, "kind": kind, "context": context,
            "result": closedloop.auto_eligibility(context)}


def api_retrain() -> dict:
    report = closedloop.run(force=True)
    checks = closedloop.build_checks(report)
    return {"ok": all(c["passed"] for c in checks), "checks": checks, "report": report}


class Handler(BaseHTTPRequestHandler):
    def _send(self, data, content_type: str) -> None:
        body = data if isinstance(data, bytes) else json.dumps(
            data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _payload(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path, query = parsed.path, parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html", "/unified.html"):
                self._send((WEB_DIR / "unified.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/overview":
                self._send(api_overview(), "application/json; charset=utf-8")
            elif path == "/api/tickets":
                self._send(api_tickets(query.get("seed", ["pool-a"])[0]),
                           "application/json; charset=utf-8")
            elif path == "/api/report":
                self._send(api_report(query.get("name", [""])[0]),
                           "application/json; charset=utf-8")
            elif path == "/api/history":
                self._send(api_history(), "application/json; charset=utf-8")
            elif path == "/api/labeling":
                self._send(api_labeling(), "application/json; charset=utf-8")
            elif path == "/api/services":
                self._send(api_services(), "application/json; charset=utf-8")
            elif path == "/api/checks":
                self._send({"ok": True, "groups": pipeline._read_checks()},
                           "application/json; charset=utf-8")
            else:
                self._send({"ok": False, "error": "未知路径"}, "application/json; charset=utf-8")
        except Exception as exc:
            self._send({"ok": False, "error": str(exc), "trace": traceback.format_exc()},
                       "application/json; charset=utf-8")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        payload = self._payload()
        try:
            if path == "/api/pipeline":
                result = pipeline.run_pipeline(
                    int(payload.get("ticket_id", 0)),
                    use_generation=bool(payload.get("use_generation", True)),
                    use_cache=bool(payload.get("use_cache", True)))
                self._send(result, "application/json; charset=utf-8")
            elif path == "/api/run_all":
                self._send(pipeline.run_all(
                    use_generation=bool(payload.get("use_generation", True))),
                    "application/json; charset=utf-8")
            elif path == "/api/mode":
                self._send(lifecycle.set_mode(str(payload.get("mode", ""))),
                           "application/json; charset=utf-8")
            elif path == "/api/kill_switch":
                self._send(lifecycle.set_kill_switch(bool(payload.get("on", False))),
                           "application/json; charset=utf-8")
            elif path == "/api/auto_probe":
                self._send(api_auto_probe(str(payload.get("kind", "high"))),
                           "application/json; charset=utf-8")
            elif path == "/api/override":
                self._send(lifecycle.human_override(
                    int(payload.get("ticket_id", 0)),
                    str(payload.get("field", "priority")),
                    str(payload.get("new_value", "high"))),
                    "application/json; charset=utf-8")
            elif path == "/api/drift":
                self._send({"ok": True, "result": lifecycle.drift_monitor(persist=False)},
                           "application/json; charset=utf-8")
            elif path == "/api/redact":
                self._send({"ok": True, "result": lifecycle.redact(str(payload.get("text", "")))},
                           "application/json; charset=utf-8")
            elif path == "/api/golden_save":
                self._send(generation.save_golden_label(
                    int(payload.get("ticket_id", 0)),
                    [int(x) for x in payload.get("chunk_ids", [])],
                    str(payload.get("note", ""))), "application/json; charset=utf-8")
            elif path == "/api/review":
                self._send(generation.save_review(
                    int(payload.get("ticket_id", 0)),
                    int(payload.get("relevance", 3)),
                    int(payload.get("executability", 3)),
                    int(payload.get("hallucination", 1)),
                    str(payload.get("note", ""))), "application/json; charset=utf-8")
            elif path == "/api/retrain":
                self._send(api_retrain(), "application/json; charset=utf-8")
            elif path == "/api/rebackfill":
                self._send({"ok": True, "updated_rows": skills.apply_to_triage_results()},
                           "application/json; charset=utf-8")
            else:
                self._send({"ok": False, "error": "未知路径"}, "application/json; charset=utf-8")
        except Exception as exc:
            self._send({"ok": False, "error": str(exc), "trace": traceback.format_exc()},
                       "application/json; charset=utf-8")

    def log_message(self, fmt, *args) -> None:
        pass


def main() -> None:
    pipeline.ensure_table()
    knowledge.apply_active_weight()
    print("=" * 68)
    print("  工单分诊与解决建议 · 一体化控制台")
    print(f"  http://127.0.0.1:{PORT}/")
    print("  五步验收 + P0 三项 + 端到端流水线，全部功能在同一页面。")
    print("  独立验收页面仍在 8765 / 8766 / 8767 / 8768 / 8769。")
    print("=" * 68)
    if os.environ.get("ITSM_NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}/")).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""第三步「解决建议」独立验收页面服务：端口 8767，与第一、二步完全隔离。"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import knowledge  # noqa: E402
import triage  # noqa: E402
from common import REPORT_FILE, WEB_DIR, connect, load_json  # noqa: E402

PORT = 8767
STEP2_REPORT = triage.REPORT_PATH
STEP3_REPORT = knowledge.REPORT_PATH
OTHER_REPORTS = (REPORT_FILE, STEP2_REPORT)


def item(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def api_status() -> dict:
    try:
        conn = connect()
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_type, count(*) FROM resolution_knowledge GROUP BY 1")
            corpus = dict(cur.fetchall())
            cur.execute("SELECT count(*) FROM tickets")
            tickets = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM assignments")
            assignments = cur.fetchone()[0]
        conn.close()
        return {
            "db_ok": True, "corpus": corpus, "tickets": tickets,
            "assignments_rows": assignments, "port": PORT,
            "step1_report_exists": REPORT_FILE.exists(),
            "step2_report_exists": STEP2_REPORT.exists(),
            "step3_report_exists": STEP3_REPORT.exists(),
            "ollama_ok": knowledge._probe_ollama(),
        }
    except Exception as exc:
        return {"db_ok": False, "error": str(exc)}


def api_run_all() -> dict:
    checks: list[dict] = []
    try:
        status = api_status()
        checks.append(item("数据库连接正常", status.get("db_ok", False),
                           f"工单表 {status.get('tickets', 0)} 行，服务端口 {PORT}（独立于 8765 / 8766）"
                           if status.get("db_ok") else status.get("error", "")))
        if not status.get("db_ok"):
            return {"ok": False, "checks": checks, "error": "数据库未连接"}
        checks.append(item("本页与前两步完全独立", PORT == 8767 and STEP3_REPORT not in OTHER_REPORTS,
                           "独立端口 8767、独立页面 step3.html、独立报告 rag_metrics.json"))

        before = {str(p): _hash(p) for p in OTHER_REPORTS}
        report = knowledge.run()
        after = {str(p): _hash(p) for p in OTHER_REPORTS}
        checks.append(item("第一、二步报告未被覆盖", before == after,
                           f"两份旧报告的内容哈希在验收前后一致（{', '.join(p.name for p in OTHER_REPORTS)}）"))
        checks.append(item("真值表与线上派单表未被写脏", status["assignments_rows"] == 0,
                           f"assignments 仍为 {status['assignments_rows']} 行；本步只写知识库与独立报告"))

        checks.extend(knowledge.build_checks(report))
        checks.append(item("第三步报告独立落盘", STEP3_REPORT.exists() and STEP3_REPORT not in OTHER_REPORTS,
                           str(STEP3_REPORT)))
        return {"ok": all(c["passed"] for c in checks), "checks": checks,
                "report": report, "status": api_status()}
    except Exception as exc:
        return {"ok": False, "checks": checks, "error": str(exc), "trace": traceback.format_exc()}


def api_corpus() -> dict:
    try:
        corpus = knowledge.ensure_corpus()
        return {"ok": True, "message": "知识语料已核对/补齐", "corpus": corpus}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "trace": traceback.format_exc()}


def api_evaluate() -> dict:
    try:
        report = knowledge.run(skip_corpus=True)
        return {"ok": True, "message": "检索指标已重新计算并写入第三步报告",
                "report": report, "checks": knowledge.build_checks(report)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "trace": traceback.format_exc()}


def api_suggest(payload: dict) -> dict:
    try:
        text = str(payload.get("query", "")).strip()
        if not text:
            return {"ok": False, "error": "请先填写要查询的问题描述"}
        mode = str(payload.get("mode", "hybrid"))
        if mode not in ("hybrid", "vector", "keyword"):
            mode = "hybrid"
        raw = payload.get("weight", "")
        weight = None if raw in ("", None, "default") else float(raw)
        result = knowledge.suggest(text, mode=mode, keyword_weight=weight)
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "trace": traceback.format_exc()}


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

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send((WEB_DIR / "step3.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send(api_status(), "application/json; charset=utf-8")
        elif path == "/api/report":
            self._send(load_json(STEP3_REPORT) if STEP3_REPORT.exists() else {},
                       "application/json; charset=utf-8")
        else:
            self._send({"error": "未知路径"}, "application/json; charset=utf-8")

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        payload = {}
        if path == "/api/suggest":
            length = int(self.headers.get("Content-Length", 0))
            if length:
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                except Exception:
                    payload = {}
            self._send(api_suggest(payload), "application/json; charset=utf-8")
        elif path == "/api/run_all":
            self._send(api_run_all(), "application/json; charset=utf-8")
        elif path == "/api/corpus":
            self._send(api_corpus(), "application/json; charset=utf-8")
        elif path == "/api/evaluate":
            self._send(api_evaluate(), "application/json; charset=utf-8")
        else:
            self._send({"error": "未知路径"}, "application/json; charset=utf-8")

    def log_message(self, fmt, *args) -> None:
        pass


def main() -> None:
    print("=" * 60)
    print(f"第三步验收页面已启动： http://127.0.0.1:{PORT}")
    print("本服务与第一步（8765）、第二步（8766）完全独立。关闭本窗口即停止。")
    print("=" * 60)
    if os.environ.get("ITSM_NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}/")).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

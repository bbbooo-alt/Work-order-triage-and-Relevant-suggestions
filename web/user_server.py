# -*- coding: utf-8 -*-
"""用户工作台服务：端口 8772。面向业务使用，不暴露验收与测试术语。"""
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import online  # noqa: E402
from common import WEB_DIR  # noqa: E402

PORT = 8772


def api_bootstrap() -> dict:
    return online.bootstrap()


def api_status() -> dict:
    return {"ok": True, "status": online.status()}


def api_history(limit: int) -> dict:
    return {"ok": True, "items": online.history(limit),
            "domains": online.skill_domains()}


def api_request(request_id: int) -> dict:
    result = online.get_request(request_id)
    if result is None:
        return {"ok": False, "error": f"找不到处理记录 {request_id}"}
    return {"ok": True, "result": result, "domains": online.skill_domains()}


def api_handle(payload: dict) -> dict:
    subject = str(payload.get("subject", ""))
    body = str(payload.get("body", ""))
    result = online.handle(subject, body,
                           use_generation=bool(payload.get("use_generation", True)),
                           use_cache=bool(payload.get("use_cache", True)))
    return {"ok": True, "result": result}


def api_dispatch(payload: dict) -> dict:
    engineer = payload.get("engineer_id")
    return online.confirm_dispatch(
        int(payload.get("request_id", 0)),
        int(engineer) if engineer not in (None, "", 0) else None,
        str(payload.get("note", "")))


def api_feedback(payload: dict) -> dict:
    rating = payload.get("rating")
    return online.save_feedback(
        int(payload.get("request_id", 0)),
        str(payload.get("field", "")), str(payload.get("value", "")),
        int(rating) if rating not in (None, "") else None,
        str(payload.get("note", "")))


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
            if path in ("/", "/index.html", "/user.html"):
                self._send((WEB_DIR / "user.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/bootstrap":
                self._send(api_bootstrap(), "application/json; charset=utf-8")
            elif path == "/api/status":
                self._send(api_status(), "application/json; charset=utf-8")
            elif path == "/api/history":
                self._send(api_history(int(query.get("limit", ["20"])[0])),
                           "application/json; charset=utf-8")
            elif path == "/api/request":
                self._send(api_request(int(query.get("id", ["0"])[0])),
                           "application/json; charset=utf-8")
            elif path == "/api/knowledge":
                self._send({"ok": True, "documents": online.knowledge_documents(),
                            "domains": online.skill_domains()},
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
            if path == "/api/handle":
                self._send(api_handle(payload), "application/json; charset=utf-8")
            elif path == "/api/dispatch":
                self._send(api_dispatch(payload), "application/json; charset=utf-8")
            elif path == "/api/feedback":
                self._send(api_feedback(payload), "application/json; charset=utf-8")
            else:
                self._send({"ok": False, "error": "未知路径"}, "application/json; charset=utf-8")
        except Exception as exc:
            self._send({"ok": False, "error": str(exc), "trace": traceback.format_exc()},
                       "application/json; charset=utf-8")

    def log_message(self, fmt, *args) -> None:
        pass


def main() -> None:
    online.ensure_tables()
    threading.Thread(target=online.warmup, daemon=True).start()
    print("=" * 68)
    print("  工单智能工作台（面向用户）")
    print(f"  http://127.0.0.1:{PORT}/")
    print("  模型在后台预热，页面顶部会显示就绪状态。")
    print("  演示环境：确认派单与反馈只写独立表，不写线上 assignments。")
    print("=" * 68)
    if os.environ.get("ITSM_NO_BROWSER") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}/")).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

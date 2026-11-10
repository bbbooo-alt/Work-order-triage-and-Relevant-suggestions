# -*- coding: utf-8 -*-
"""第四步「路由到工程师」独立验收页面服务：端口 8768，与前三步完全隔离。"""
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
import route  # noqa: E402
import triage  # noqa: E402
from common import REPORT_FILE, WEB_DIR, connect, load_json  # noqa: E402

PORT = 8768
STEP2_REPORT = triage.REPORT_PATH
STEP3_REPORT = knowledge.REPORT_PATH
STEP4_REPORT = route.REPORT_PATH
OTHER_REPORTS = (REPORT_FILE, STEP2_REPORT, STEP3_REPORT)
_CACHE: dict = {}


def item(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _context() -> tuple[dict, list[dict]]:
    if "static" not in _CACHE:
        conn = connect()
        _CACHE["static"] = route.load_static(conn)
        _CACHE["replay"] = route.load_replay(conn, _CACHE["static"])
        conn.close()
    return _CACHE["static"], _CACHE["replay"]


def api_status() -> dict:
    try:
        conn = connect()
        with conn.cursor() as cur:
            cur.execute("""SELECT model_version, count(*) FROM route_assignments
                           GROUP BY 1 ORDER BY 1""")
            route_rows = dict(cur.fetchall())
            cur.execute("SELECT count(*) FROM assignments")
            online = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM eval_assignments WHERE model_version = 'round-robin-v0'")
            baseline = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM tickets")
            tickets = cur.fetchone()[0]
        conn.close()
        return {"db_ok": True, "tickets": tickets, "route_rows": route_rows,
                "online_assignments_rows": online, "baseline_replay_rows": baseline,
                "port": PORT,
                "step1_report_exists": REPORT_FILE.exists(),
                "step2_report_exists": STEP2_REPORT.exists(),
                "step3_report_exists": STEP3_REPORT.exists(),
                "step4_report_exists": STEP4_REPORT.exists()}
    except Exception as exc:
        return {"db_ok": False, "error": str(exc)}


def api_run_all() -> dict:
    checks: list[dict] = []
    try:
        status = api_status()
        checks.append(item("数据库连接正常", status.get("db_ok", False),
                           f"工单表 {status.get('tickets', 0)} 行，服务端口 {PORT}（独立于 8765 / 8766 / 8767）"
                           if status.get("db_ok") else status.get("error", "")))
        if not status.get("db_ok"):
            return {"ok": False, "checks": checks, "error": "数据库未连接"}
        checks.append(item("本页与前三步完全独立", PORT == 8768 and STEP4_REPORT not in OTHER_REPORTS,
                           "独立端口 8768、独立页面 step4.html、独立报告 routing_metrics.json"))

        before = {str(p): _hash(p) for p in OTHER_REPORTS}
        report = route.run()
        after = {str(p): _hash(p) for p in OTHER_REPORTS}
        checks.append(item("前三步报告未被覆盖", before == after,
                           f"三份旧报告内容哈希在验收前后一致（{', '.join(p.name for p in OTHER_REPORTS)}）"))
        _CACHE.pop("static", None)
        _CACHE.pop("replay", None)
        checks.extend(route.build_checks(report))
        checks.append(item("第四步报告独立落盘", STEP4_REPORT.exists(), str(STEP4_REPORT)))
        return {"ok": all(c["passed"] for c in checks), "checks": checks,
                "report": report, "status": api_status()}
    except Exception as exc:
        return {"ok": False, "checks": checks, "error": str(exc), "trace": traceback.format_exc()}


def api_replay() -> dict:
    try:
        report = route.run()
        _CACHE.pop("static", None)
        _CACHE.pop("replay", None)
        return {"ok": True, "message": "派单离线回放已重跑（含并发压测与幂等检查）",
                "report": report, "checks": route.build_checks(report)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "trace": traceback.format_exc()}


def api_metrics() -> dict:
    try:
        if not STEP4_REPORT.exists():
            return {"ok": False, "error": "尚未回放，请先点「一键验收第四步」"}
        report = load_json(STEP4_REPORT)
        return {"ok": True, "message": "已从落盘报告重新计算检查项（未重新派单）",
                "report": report, "checks": route.build_checks(report)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "trace": traceback.format_exc()}


def api_tickets() -> dict:
    try:
        _, replay = _context()
        conn = connect()
        cur = conn.cursor()
        cur.execute("""SELECT ticket_id, status FROM route_assignments
                       WHERE model_version = %s""", (route.MODEL_VERSION,))
        status = dict(cur.fetchall())
        conn.close()
        assigned = [t for t in replay if status.get(t["ticket_id"]) == "assigned"][:5]
        degraded = [t for t in replay if status.get(t["ticket_id"]) == "manual_queue"][:3]
        pack = lambda t: {"ticket_id": t["ticket_id"], "priority": t["priority"],
                          "pred_priority": t.get("pred_priority"),
                          "required_skills": sorted(t["required"]),
                          "expected": status.get(t["ticket_id"], "unknown")}
        return {"ok": True, "tickets": [pack(t) for t in assigned + degraded]}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "trace": traceback.format_exc()}


def api_explain(payload: dict) -> dict:
    try:
        static, replay = _context()
        ticket_id = int(payload.get("ticket_id", 0))
        return route.explain(ticket_id, static, replay)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "trace": traceback.format_exc()}


def api_idempotency() -> dict:
    try:
        static, replay = _context()
        return {"ok": True, "result": route.check_idempotency(static, replay)}
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

    def _payload(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send((WEB_DIR / "step4.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send(api_status(), "application/json; charset=utf-8")
        elif path == "/api/report":
            self._send(load_json(STEP4_REPORT) if STEP4_REPORT.exists() else {},
                       "application/json; charset=utf-8")
        elif path == "/api/tickets":
            self._send(api_tickets(), "application/json; charset=utf-8")
        else:
            self._send({"error": "未知路径"}, "application/json; charset=utf-8")

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if path == "/api/run_all":
            self._send(api_run_all(), "application/json; charset=utf-8")
        elif path == "/api/replay":
            self._send(api_replay(), "application/json; charset=utf-8")
        elif path == "/api/metrics":
            self._send(api_metrics(), "application/json; charset=utf-8")
        elif path == "/api/explain":
            self._send(api_explain(self._payload()), "application/json; charset=utf-8")
        elif path == "/api/idempotency":
            self._send(api_idempotency(), "application/json; charset=utf-8")
        else:
            self._send({"error": "未知路径"}, "application/json; charset=utf-8")

    def log_message(self, fmt, *args) -> None:
        pass


def main() -> None:
    print("=" * 60)
    print(f"第四步验收页面已启动： http://127.0.0.1:{PORT}")
    print("本服务与前三步（8765 / 8766 / 8767）完全独立。关闭本窗口即停止。")
    print("=" * 60)
    if os.environ.get("ITSM_NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}/")).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

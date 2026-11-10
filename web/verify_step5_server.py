# -*- coding: utf-8 -*-
"""第五步「上线闭环」独立验收页面服务：端口 8769，与前四步完全隔离。"""
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
import lifecycle  # noqa: E402
import route  # noqa: E402
import triage  # noqa: E402
from common import REPORT_FILE, WEB_DIR, connect, load_json  # noqa: E402

PORT = 8769
STEP2_REPORT = triage.REPORT_PATH
STEP3_REPORT = knowledge.REPORT_PATH
STEP4_REPORT = route.REPORT_PATH
STEP5_REPORT = lifecycle.REPORT_PATH
OTHER_REPORTS = (REPORT_FILE, STEP2_REPORT, STEP3_REPORT, STEP4_REPORT)


def item(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def api_status() -> dict:
    try:
        conn = connect()
        cur = conn.cursor()
        counts = {}
        for table in ("decision_log", "feedback_events", "drift_snapshots", "auto_assignments"):
            cur.execute(f"SELECT count(*) FROM {table}")
            counts[table] = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM assignments")
        online = cur.fetchone()[0]
        cur.execute("SELECT split, count(*) FROM tickets GROUP BY 1")
        splits = dict(cur.fetchall())
        conn.close()
        state = lifecycle.mode_state()
        return {"db_ok": True, "port": PORT, "state": state, "counts": counts,
                "online_assignments_rows": online, "splits": splits,
                "step1_report_exists": REPORT_FILE.exists(),
                "step2_report_exists": STEP2_REPORT.exists(),
                "step3_report_exists": STEP3_REPORT.exists(),
                "step4_report_exists": STEP4_REPORT.exists(),
                "step5_report_exists": STEP5_REPORT.exists()}
    except Exception as exc:
        return {"db_ok": False, "error": str(exc)}


def api_run_all() -> dict:
    checks: list[dict] = []
    try:
        status = api_status()
        checks.append(item("数据库连接正常", status.get("db_ok", False),
                           f"工单 test {status.get('splits', {}).get('test', 0)} 行，服务端口 {PORT}"
                           "（独立于 8765 / 8766 / 8767 / 8768）"
                           if status.get("db_ok") else status.get("error", "")))
        if not status.get("db_ok"):
            return {"ok": False, "checks": checks, "error": "数据库未连接"}
        checks.append(item("本页与前四步完全独立", PORT == 8769 and STEP5_REPORT not in OTHER_REPORTS,
                           "独立端口 8769、独立页面 step5.html、独立报告 lifecycle_metrics.json"))

        before = {str(p): _hash(p) for p in OTHER_REPORTS}
        report = lifecycle.run()
        after = {str(p): _hash(p) for p in OTHER_REPORTS}
        checks.append(item("前四步报告未被覆盖", before == after,
                           f"四份旧报告内容哈希在验收前后一致（{', '.join(p.name for p in OTHER_REPORTS)}）"))
        checks.extend(lifecycle.build_checks(report))
        checks.append(item("第五步报告独立落盘", STEP5_REPORT.exists(), str(STEP5_REPORT)))
        return {"ok": all(c["passed"] for c in checks), "checks": checks,
                "report": report, "status": api_status()}
    except Exception as exc:
        return {"ok": False, "checks": checks, "error": str(exc), "trace": traceback.format_exc()}


def api_metrics() -> dict:
    try:
        if not STEP5_REPORT.exists():
            return {"ok": False, "error": "尚未演练，请先点「一键验收第五步」"}
        report = load_json(STEP5_REPORT)
        return {"ok": True, "message": "已从落盘报告重新计算检查项（未重新演练）",
                "report": report, "checks": lifecycle.build_checks(report)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "trace": traceback.format_exc()}


def api_tickets() -> dict:
    try:
        conn = connect()
        cur = conn.cursor()
        cur.execute("""(SELECT ticket_id, split, priority FROM tickets WHERE split = 'train'
                        ORDER BY md5(ticket_id::text || 'fb') LIMIT 4)
                       UNION ALL
                       (SELECT ticket_id, split, priority FROM tickets WHERE split = 'test'
                        ORDER BY md5(ticket_id::text || 'fb') LIMIT 4)
                       ORDER BY split, ticket_id""")
        rows = cur.fetchall()
        conn.close()
        return {"ok": True, "tickets": [{"ticket_id": r[0], "split": r[1],
                                         "priority": r[2]} for r in rows]}
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
            self._send((WEB_DIR / "step5.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send(api_status(), "application/json; charset=utf-8")
        elif path == "/api/report":
            self._send(load_json(STEP5_REPORT) if STEP5_REPORT.exists() else {},
                       "application/json; charset=utf-8")
        elif path == "/api/tickets":
            self._send(api_tickets(), "application/json; charset=utf-8")
        else:
            self._send({"error": "未知路径"}, "application/json; charset=utf-8")

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        payload = self._payload() if path != "/api/run_all" and path != "/api/metrics" else {}
        try:
            if path == "/api/run_all":
                self._send(api_run_all(), "application/json; charset=utf-8")
            elif path == "/api/metrics":
                self._send(api_metrics(), "application/json; charset=utf-8")
            elif path == "/api/mode":
                self._send(lifecycle.set_mode(str(payload.get("mode", ""))),
                           "application/json; charset=utf-8")
            elif path == "/api/kill_switch":
                self._send(lifecycle.set_kill_switch(bool(payload.get("on", False))),
                           "application/json; charset=utf-8")
            elif path == "/api/attempt_auto":
                self._send({"ok": True, "result": lifecycle.attempt_auto()},
                           "application/json; charset=utf-8")
            elif path == "/api/override":
                self._send(lifecycle.human_override(
                    int(payload.get("ticket_id", 0)), str(payload.get("field", "priority")),
                    str(payload.get("new_value", "high"))), "application/json; charset=utf-8")
            elif path == "/api/drift":
                self._send({"ok": True, "result": lifecycle.drift_monitor()},
                           "application/json; charset=utf-8")
            elif path == "/api/redact":
                self._send({"ok": True, "result": lifecycle.redact(
                    str(payload.get("text", "")))}, "application/json; charset=utf-8")
            else:
                self._send({"error": "未知路径"}, "application/json; charset=utf-8")
        except Exception as exc:
            self._send({"ok": False, "error": str(exc), "trace": traceback.format_exc()},
                       "application/json; charset=utf-8")

    def log_message(self, fmt, *args) -> None:
        pass


def main() -> None:
    lifecycle.ensure_tables()
    print("=" * 60)
    print(f"第五步验收页面已启动： http://127.0.0.1:{PORT}")
    print("本服务与前四步（8765 / 8766 / 8767 / 8768）完全独立。关闭本窗口即停止。")
    print("=" * 60)
    if os.environ.get("ITSM_NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}/")).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""第二步「分诊」独立验收页面服务：端口 8766，与第一步完全隔离。"""
from __future__ import annotations

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

import evaluate  # noqa: E402
import freeze_split  # noqa: E402
import triage  # noqa: E402
from common import (BASELINE_MODELS, REPORT_FILE, WEB_DIR,  # noqa: E402
                    connect, load_json)

PORT = 8766
STEP2_REPORT = triage.REPORT_PATH


def item(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def api_status() -> dict:
    try:
        conn = connect()
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM tickets")
            tickets = cur.fetchone()[0]
            cur.execute("SELECT split, count(*) FROM tickets GROUP BY 1")
            splits = dict(cur.fetchall())
            cur.execute("SELECT count(*) FROM assignments")
            assignments_rows = cur.fetchone()[0]
            cur.execute("SELECT model_version, count(*) FROM triage_results GROUP BY 1 ORDER BY 1")
            predictions = dict(cur.fetchall())
        conn.close()
        return {
            "db_ok": True,
            "tickets": tickets,
            "splits": splits,
            "predictions": predictions,
            "assignments_rows": assignments_rows,
            "step1_report_exists": REPORT_FILE.exists(),
            "step2_report_exists": STEP2_REPORT.exists(),
            "port": PORT,
        }
    except Exception as exc:
        return {"db_ok": False, "error": str(exc)}


def api_run_all() -> dict:
    checks: list[dict] = []
    try:
        status = api_status()
        checks.append(item("数据库连接正常", status.get("db_ok", False),
                           f"工单表 {status.get('tickets', 0)} 行，服务端口 {PORT}（独立于第一步）"))
        if not status.get("db_ok"):
            return {"ok": False, "checks": checks, "error": "数据库未连接"}

        frozen = freeze_split.check()
        checks.append(item("冻结的数据划分未被改动", frozen["passed"],
                           f"冻结清单 {frozen['frozen_rows']} 行，与数据库不一致 {frozen['mismatch_rows']} 行"))

        step1 = load_json(REPORT_FILE) if REPORT_FILE.exists() else {}
        step1_ok = all(m in step1.get("models", {}) for m in BASELINE_MODELS)
        checks.append(item("第一步产物完好未被覆盖", step1_ok,
                           "baseline_metrics.json 仍包含多数类与关键词规则两条基线"))

        meta = triage.train_and_predict()

        first = evaluate.run(extra_models=[triage.MODEL_VERSION], report_path=STEP2_REPORT)
        second = evaluate.run(extra_models=[triage.MODEL_VERSION], report_path=STEP2_REPORT)
        strip = lambda d: {k: v for k, v in d.items() if k != "generated_at"}  # noqa: E731
        checks.append(item("重复评测结果完全一致", strip(first) == strip(second),
                           "连续运行统一评测脚本两次，指标逐项相同"))

        report = triage.evaluate_triage(meta)
        checks.extend(triage.build_checks(report))
        checks.append(item("第二步报告已独立落盘", STEP2_REPORT.exists(),
                           str(STEP2_REPORT) + "（未写入第一步报告文件）"))

        return {
            "ok": all(c["passed"] for c in checks),
            "checks": checks,
            "report": report,
            "status": api_status(),
        }
    except Exception as exc:
        return {"ok": False, "checks": checks, "error": str(exc), "trace": traceback.format_exc()}


def api_train() -> dict:
    try:
        meta = triage.train_and_predict()
        return {"ok": True, "message": "分诊模型已重新训练并写入预测", "meta": meta}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "trace": traceback.format_exc()}


def api_evaluate() -> dict:
    try:
        report = load_json(STEP2_REPORT) if STEP2_REPORT.exists() else {}
        meta = report.get("triage")
        if not meta:
            return {"ok": False, "error": "尚未训练，请先点「一键验收第二步」"}
        return {"ok": True, "message": "指标已重新计算", "report": triage.evaluate_triage(meta)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "trace": traceback.format_exc()}


class Handler(BaseHTTPRequestHandler):
    def _send(self, data, content_type: str) -> None:
        body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send((WEB_DIR / "step2.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send(api_status(), "application/json; charset=utf-8")
        elif path == "/api/report":
            self._send(load_json(STEP2_REPORT) if STEP2_REPORT.exists() else {},
                       "application/json; charset=utf-8")
        else:
            self._send({"error": "未知路径"}, "application/json; charset=utf-8")

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if path == "/api/run_all":
            self._send(api_run_all(), "application/json; charset=utf-8")
        elif path == "/api/train":
            self._send(api_train(), "application/json; charset=utf-8")
        elif path == "/api/evaluate":
            self._send(api_evaluate(), "application/json; charset=utf-8")
        else:
            self._send({"error": "未知路径"}, "application/json; charset=utf-8")

    def log_message(self, fmt, *args) -> None:
        pass


def main() -> None:
    print("=" * 60)
    print(f"第二步验收页面已启动： http://127.0.0.1:{PORT}")
    print("本服务与第一步（端口 8765）完全独立。关闭本窗口即停止。")
    print("=" * 60)
    if os.environ.get("ITSM_NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}/")).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

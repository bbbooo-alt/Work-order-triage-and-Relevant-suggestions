# -*- coding: utf-8 -*-
"""第一步「评测地基」验收页的本地服务，只监听 127.0.0.1，启动后自动打开浏览器。"""
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
import run_baseline  # noqa: E402
from common import (BASELINE_MODELS, REPORT_FILE, SPLIT_FILE,  # noqa: E402
                    THRESHOLD_FILE, WEB_DIR, connect, load_json)

PORT = 8765


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
            cur.execute("SELECT count(*) FROM eval_assignments")
            eval_rows = cur.fetchone()[0]
            cur.execute("""SELECT model_version, count(*) FROM triage_results
                           WHERE model_version = ANY(%s) GROUP BY 1""", (BASELINE_MODELS,))
            predictions = dict(cur.fetchall())
        conn.close()
        return {
            "db_ok": True,
            "tickets": tickets,
            "splits": splits,
            "baseline_predictions": predictions,
            "assignments_rows": assignments_rows,
            "eval_assignments_rows": eval_rows,
            "report_exists": REPORT_FILE.exists(),
            "thresholds_exists": THRESHOLD_FILE.exists(),
            "split_file_exists": SPLIT_FILE.exists(),
        }
    except Exception as exc:
        return {"db_ok": False, "error": str(exc)}


def api_run_all() -> dict:
    checks: list[dict] = []
    try:
        status = api_status()
        checks.append(item("数据库连接正常", status.get("db_ok", False),
                           f"工单表 {status.get('tickets', 0)} 行" if status.get("db_ok") else status.get("error", "")))
        if not status.get("db_ok"):
            return {"ok": False, "checks": checks, "error": "数据库未连接，请先确认本地 PostgreSQL 已启动"}

        frozen = freeze_split.run()
        counts = frozen["counts"]
        checks.append(item("冻结数据划分（train/val/test）", frozen["passed"],
                           f"共 {frozen['total']} 行：train {counts['train']} / val {counts['val']} / test {counts['test']}；"
                           f"同一正文跨集 {frozen['cross_split_same_body']} 条"))
        consistency = freeze_split.check()
        checks.append(item("冻结清单与数据库一致", consistency["passed"],
                           f"清单 {consistency['frozen_rows']} 行，不一致 {consistency['mismatch_rows']} 行"))

        baseline = run_baseline.run()
        expected = baseline["test_tickets"] * 2
        checks.append(item("两条基线预测已生成", baseline["prediction_rows"] == expected,
                           f"{baseline['prediction_rows']} 条预测（多数类 + 关键词规则，各 {baseline['test_tickets']} 条）"))

        first = evaluate.run()
        second = evaluate.run()
        a = {k: v for k, v in first.items() if k != "generated_at"}
        b = {k: v for k, v in second.items() if k != "generated_at"}
        checks.append(item("重复评测结果完全一致", a == b,
                           "连续运行两次评测，所有指标逐项相同（固定随机种子）"))

        tasks = ("type", "queue", "priority")
        complete = all(task in second["models"][m] and second["models"][m][task]["macro_f1"] is not None
                       for m in BASELINE_MODELS for task in tasks)
        checks.append(item("三类任务指标齐全", complete,
                           "工单类型 / 处理部门 / 优先级，均含 macro-F1、逐类 F1、混淆矩阵"))

        keyword = second["models"]["keyword-v0"]["type"]["macro_f1"]
        majority = second["models"]["majority-v0"]["type"]["macro_f1"]
        checks.append(item("基线对照有效", keyword > majority,
                           f"关键词规则 {keyword} 高于多数类瞎猜 {majority}（工单类型 macro-F1）"))

        isolation = second["isolation"]
        checks.append(item("线上派单表未被污染", isolation["assignments_rows"] == 0,
                           f"assignments 仍为 {isolation['assignments_rows']} 行；评测数据写在独立表 eval_assignments"))
        checks.append(item("预测未越出测试集", isolation["stray_predictions_outside_test"] == 0,
                           f"越界预测 {isolation['stray_predictions_outside_test']} 条"))

        thresholds = load_json(THRESHOLD_FILE) if THRESHOLD_FILE.exists() else {}
        checks.append(item("进阶门槛已写死", "min_macro_f1_gain_over_keyword_baseline" in thresholds,
                           f"后续模型 macro-F1 至少 +{thresholds.get('min_macro_f1_gain_over_keyword_baseline', '?')}，且无类别下降"))
        checks.append(item("评测报告已落盘", REPORT_FILE.exists(), str(REPORT_FILE)))

        return {
            "ok": all(c["passed"] for c in checks),
            "checks": checks,
            "report": second,
            "status": api_status(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "checks": checks,
            "error": str(exc),
            "trace": traceback.format_exc(),
        }


def api_run_baseline() -> dict:
    try:
        result = run_baseline.run()
        return {"ok": True, "message": "基线已重新生成", "result": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "trace": traceback.format_exc()}


def api_evaluate() -> dict:
    try:
        return {"ok": True, "message": "指标已重新计算", "report": evaluate.run()}
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
            self._send((WEB_DIR / "step1.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send(api_status(), "application/json; charset=utf-8")
        elif path == "/api/report":
            self._send(load_json(REPORT_FILE) if REPORT_FILE.exists() else {},
                       "application/json; charset=utf-8")
        else:
            self._send({"error": "未知路径"}, "application/json; charset=utf-8")

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if path == "/api/run_all":
            self._send(api_run_all(), "application/json; charset=utf-8")
        elif path == "/api/run_baseline":
            self._send(api_run_baseline(), "application/json; charset=utf-8")
        elif path == "/api/evaluate":
            self._send(api_evaluate(), "application/json; charset=utf-8")
        else:
            self._send({"error": "未知路径"}, "application/json; charset=utf-8")

    def log_message(self, fmt, *args) -> None:
        pass


def main() -> None:
    print("=" * 60)
    print("第一步验收页面已启动： http://127.0.0.1:" + str(PORT))
    print("浏览器会自动打开；使用完毕直接关闭本窗口即可。")
    print("=" * 60)
    if os.environ.get("ITSM_NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}/")).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

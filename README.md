# 工单分诊与解决建议

> ITSM Ticket Triage & Resolution Assistant

一个**本机离线验证型原型**：把工单从接入到处置的整条链路跑通，并且每一步都可复现、可验收。链路覆盖工单接入与 PII 脱敏、分诊、技能多标签预测、混合检索与拒答、基于片段的生成建议、技能∩排班∩容量路由、置信度＋风险自动档门禁，以及闭环回流与重训。

**这不是生产系统**：不含真实业务数据、没有部署配置、默认影子模式只记录建议不自动派单；请勿用于真实派单或任何商业用途。

---

## 目录

1. [项目定位与设计原则](#1-项目定位与设计原则)
2. [页面地图：先打开哪一个](#2-页面地图先打开哪一个)
3. [技术栈](#3-技术栈)
4. [快速开始](#4-快速开始)
5. [目录结构](#5-目录结构)
6. [环境变量](#6-环境变量)
7. [评测、报告与验收](#7-评测报告与验收)
8. [数据隔离与安全边界](#8-数据隔离与安全边界)
9. [许可证与合规](#9-许可证与合规)
10. [已知限制与上传 GitHub 前检查清单](#10-已知限制与上传-github-前检查清单)
11. [发布到 GitHub](#11-发布到-github)
12. [常见问题](#12-常见问题)
13. [延伸阅读](#13-延伸阅读)

---

## 1. 项目定位与设计原则

七段链路（`scripts/pipeline.py` 中的 `STAGE_DEFS`）：

| 段 | 内容 |
| --- | --- |
| ① `intake` | 工单接入与 PII 脱敏 |
| ② `triage` | 分诊预测（类型 / 部门 / 优先级 / 置信度） |
| ③ `skills` | 技能多标签预测（P0-1） |
| ④ `retrieval` | 混合检索与拒答（第三步） |
| ⑤ `generation` | 基于检索片段的生成（P0-2） |
| ⑥ `routing` | 技能 ∩ 排班 ∩ 容量路由（第四步） |
| ⑦ `gate` | 置信度＋风险自动档门禁（P0-3 / 第五步） |

贯穿全项目的几条约束：

- **评测先行**：`train/val/test` 划分一次性冻结在 `eval/split_frozen.csv`，`test` 只评一次；先跑关键词＋轮询基线，再谈模型收益。
- **真值与预测分离**：`tickets` 存标注真值，`triage_results` / `skill_predictions` 存预测，评测绝不混用。
- **隔离写库**：验收、回放、流水线、在线请求各自写独立表，生产 `assignments` 恒为 0 行。
- **可解释与可审计**：派单写 `reason`；自动档门禁逐项给出 `kill_switch_off / confidence_ok / risk_low / retrieval_grounded / routing_assigned`。
- **护栏**：检索有 `abstain` 拒答与 grounding 校验；全局 `shadow / assist / auto` 与单据级风险门禁两层控制。
- **报告哈希保护**：一体化全量验收前后比对 11 份既有报告哈希，确保验收过程不污染既有产物。

---

## 2. 页面地图：先打开哪一个

全部页面都是本机 `127.0.0.1` 的独立 HTTP 服务，互不依赖，也不需要同时启动。

| 端口 | 面向 | 页面 / 启动脚本 | 用途 |
| --- | --- | --- | --- |
| **8772** | **业务使用** | `启动项/启动用户工作台.bat` → `web/user.html` | 提交任意工单，得到分诊、知识依据、处理建议、派单推荐、处置结论；含人工复核与反馈 |
| **8771** | 开发 / 测试 | `启动项/启动分步验收台.bat` → `web/testbench.html` | 按流程分组的 **140 个可单独点击的测试点**，每点写明「测什么 / 怎么测 / 期望」，右侧输出结构化日志；另有冒烟、回归、全量顺序执行 |
| **8770** | 开发 / 验收 | `启动项/启动一体化页面.bat` → `web/unified.html` | 五步验收 + 3 项 P0 + 端到端流水线融在一页，`/api/run_all` 一键全量验收 |
| 8765 | 第一步 | `启动项/启动验收页面_第一步.bat` → `web/step1.html` | 评测地基 / 基线 |
| 8766 | 第二步 | `启动项/启动验收页面_第二步.bat` → `web/step2.html` | 分诊 |
| 8767 | 第三步 | `启动项/启动验收页面_第三步.bat` → `web/step3.html` | 解决建议 / RAG |
| 8768 | 第四步 | `启动项/启动验收页面_第四步.bat` → `web/step4.html` | 路由到工程师 |
| 8769 | 第五步 | `启动项/启动验收页面_第五步.bat` → `web/step5.html` | 上线闭环 / 自动档门禁 |

> 只想「用」项目 → 开 8772；想知道「每个点到底测了什么」 → 开 8771；想看「连起来跑的结果」 → 开 8770。
>
> 所有 `.bat` 已统一从项目根目录运行，环境检查集中在 `启动项/_env_check.bat`；也可以直接在项目根目录运行 `python web/verify_server.py` 等命令。本机 PostgreSQL 路径配置见[第 4.7 节](#47-启动页面)。

---

## 3. 技术栈

| 层 | 选型 |
| --- | --- |
| 语言 | Python **3.10+**（用到 `str \| None` 等类型标注） |
| 数据库 | PostgreSQL 17 + **pgvector 0.8.0**（`vector(1024)` + HNSW 余弦索引） |
| 模型服务 | 本地 **Ollama** `http://127.0.0.1:11434` |
| 嵌入模型 | `qwen3-embedding:4b`，原生 2,560 维 → 按 Matryoshka 前缀截断到 **1,024 维**（HNSW 上限 2,000 维） |
| 生成模型 | `qwen3:14b`（首次调用较慢，约十几秒） |
| 分类模型 | `TfidfVectorizer(char_wb, 3-5gram)` + `SGDClassifier(log_loss)`；L1 分诊不依赖 Ollama |
| Web | Python 标准库 `http.server.ThreadingHTTPServer` + 原生 HTML/CSS/JS，**无 Web 框架** |
| 数据规模 | 28,587 条工单、13 个技能域、12 篇知识文档、12 位造数工程师 |

数据库以 `sql/01_schema.sql` 的 8 张基础表起步（`skill_domains`、`engineers`、`tickets`、`resolution_knowledge`、`engineer_skills`、`shifts`、`triage_results`、`assignments`），运行时各模块还会按需创建评测、路由、生成缓存、闭环与用户工作台等表。**实际表结构已超出最初 8 表设计**，以 `文档/项目现状与计划实施偏差复盘.md` 为准。

---

## 4. 快速开始

### 4.1 前置条件

- Windows（`.bat` 启动脚本与 Ollama 可执行文件定位均为 Windows 专属）
- Python 3.10+
- PostgreSQL 17，已安装并可启用 `vector` 扩展，监听 `127.0.0.1:5432`
- Ollama，已拉取 `qwen3-embedding:4b` 与 `qwen3:14b`

### 4.2 安装 Python 依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4.3 准备 PostgreSQL

启动本机 PostgreSQL，确认扩展可用（项目脚本 `sql/01_schema.sql` 第一行会执行 `CREATE EXTENSION IF NOT EXISTS vector`，但扩展本体需先在服务器上安装）：

```powershell
pg_isready -h 127.0.0.1 -p 5432
psql -h 127.0.0.1 -U postgres -d postgres -c "SELECT extversion FROM pg_extension WHERE extname='vector';"
```

连接参数集中在 `scripts/common.py`，并在 `scripts/etl.py`、`scripts/verify.py`、`scripts/embed_knowledge.py` 中重复了一份；默认库名 `itsm_triage`、用户 `postgres`、**不传密码**（依赖本机 trust 免密认证）。若你的 PG 需要密码，请设置 `PGPASSWORD` 环境变量或配置 `pgpass.conf`，由 libpq 自动读取；不要把密码写进代码。

### 4.4 准备 Ollama

```powershell
ollama pull qwen3-embedding:4b
ollama pull qwen3:14b
ollama serve
python -c "import sys; sys.path.insert(0,'scripts'); import knowledge; print(knowledge.ensure_ollama())"
```

### 4.5 准备数据集（必读）

`scripts/etl.py` 读取固定的原始文件：

```
data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv
```

该文件为 28,587 行英/德文客户支持工单，许可证 **`cc-by-nc-4.0`（禁止商用）**，且项目文档未记录其出处/下载链接。**本仓库通过 `.gitignore` 排除了 `data/raw/` 与 `data/processed/`，不随仓库分发数据**；请自行合法获取并放置到上述路径，然后由 `etl.py` 重建 `data/processed/` 中间产物。

工程师、技能等级、排班、12 篇知识库文档均为**演示造数**，不是真实业务数据。

### 4.6 初始化与全量验收（顺序不可颠倒）

```powershell
# 1) 建库 → 建表 → CSV 清洗/造数 → 导入 28,587 行 → 建索引 → 自检
python scripts/etl.py

# 2) 灌入 1,024 维向量（默认只灌 3000 条；全量用 $env:ITSM_EMBED_LIMIT="0"）
python scripts/embed_knowledge.py

# 3) 只读自检：扩展版本、HNSW 索引、向量维度、余弦检索样例、PII 残留
python scripts/verify.py

# 4) 固定划分（仓库已附 eval/split_frozen.csv，通常无需重跑；重跑会改变评测基线）
python scripts/freeze_split.py

# 5) 一键全量验收，生成 reports/pipeline_metrics.json
python scripts/pipeline.py
```

`pipeline.run_all()` 会编排各阶段并做报告哈希前后比对；若只想看单项结果，可单独运行 `scripts/` 下的 `run_baseline.py`、`triage.py`、`skills.py`、`knowledge.py`、`generation.py`、`route.py`、`lifecycle.py`、`closedloop.py`。更细的检查请用 8771 分步验收台。

### 4.7 启动页面

两种方式任选：

```powershell
# A) 运行启动脚本（也可直接双击 .bat）
.\启动项\启动用户工作台.bat

# B) 直接在项目根目录运行 Python 服务
python web/user_server.py        # 8772 用户工作台
python web/testbench_server.py   # 8771 分步验收台
python web/unified_server.py     # 8770 一体化控制台
```

`.bat` 先调用 `启动项/_env_check.bat` 检查 PostgreSQL（需要 Ollama 的页面额外检查 Ollama），统一切到项目根目录后再启动服务。**仓库里不含本机路径**：若 PG 已启动且 `pg_isready` 在 PATH 上，无需任何配置；若需要脚本自动拉起 PG，把 `启动项/local_env.example.bat` 复制为 `启动项/local_env.bat`（已被 `.gitignore` 排除），填写 `PG_BIN`（含 `pg_isready.exe` / `pg_ctl.exe` 的目录）与 `PG_DATA`（数据目录）。

服务会自动打开浏览器；设置 `ITSM_NO_BROWSER=1` 可关闭自动打开。

---

## 5. 目录结构

```
.
├── README.md                  # 本文件
├── requirements.txt           # Python 依赖
├── .env.example               # 环境变量清单（仅作参考，不自动加载）
├── .gitignore / .gitattributes
├── 启动项/                    # 8 个 Windows 启动脚本
│   ├── _env_check.bat         # 公共环境检查（PG / Ollama）
│   ├── local_env.example.bat  # 本机 PG 路径配置示例
│   └── local_env.bat          # 本机配置（gitignore，不入库）
├── scripts/                   # ETL、评测、分诊、技能、检索、生成、路由、闭环、流水线、在线推理
│   ├── common.py              # 路径 / PG 连接 / JSON 工具
│   ├── etl.py                 # 初始化入口
│   ├── pipeline.py            # 七段链路编排 + 全量验收
│   └── online.py              # 8772 在线推理层（自由文本、dry_run、隔离写库）
├── web/                       # 8 个 HTTP 服务 + 8 个前端页面
├── sql/                       # 01_schema.sql（建表）、02_indexes.sql（索引）
├── eval/                      # 阈值、校准、冻结划分、黄金集、知识库文档
├── reports/                   # 11 份 JSON 报告产物（随仓库保留，作为验收证据）
├── data/raw/                  # 原始 CSV（被忽略，需自行准备）
├── data/processed/            # 中间 CSV（被忽略，由 etl.py 生成）
├── 文档/                      # 4 篇正式文档
└── 思路描述/                  # 5 篇分步设计文档
```

---

## 6. 环境变量

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `ITSM_DB` | `itsm_triage` | 数据库名 |
| `PGHOST` / `PGPORT` / `PGUSER` | `127.0.0.1` / `5432` / `postgres` | PostgreSQL 连接 |
| `PGPASSWORD` | — | 由 libpq 自动读取；代码不传密码 |
| `PG_BIN` / `PG_DATA` | — | `.bat` 专用，写在 `启动项/local_env.bat`；用于自动拉起本机 PG |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama 地址 |
| `EMBED_MODEL` | `qwen3-embedding:4b` | 嵌入模型 |
| `EMBED_DIM` | `1024` | 前缀截断维度，须与 `vector(1024)` 一致 |
| `EMBED_BATCH` | `16` | 灌向量批大小 |
| `ITSM_EMBED_LIMIT` | `3000` | 灌向量条数，`0` 表示全量 |
| `ITSM_NO_BROWSER` | `0` | 设为 `1` 禁止服务自动打开浏览器 |
| `LOCALAPPDATA` | 系统提供 | Windows 下用于定位 `ollama.exe` |

> **`.env.example` 只是清单**：项目没有使用 `python-dotenv`，代码不会读取 `.env`。请在启动前用 shell 设置，例如 PowerShell `$env:ITSM_DB="itsm_triage"`，或 CMD `set ITSM_DB=itsm_triage`。
>
> `PG_BIN` / `PG_DATA` 属于 `.bat` 的本机配置，写在 `启动项/local_env.bat`（见[第 4.7 节](#47-启动页面)），不是 Python 环境变量。

---

## 7. 评测、报告与验收

- `eval/thresholds.json`：阶段准入门槛（macro-F1 相对关键词基线 ≥ +0.03、路由成功率 ≥ 0.90、超时率 ≤ 0.20）。
- `eval/calibration.json`：自动档置信度阈值（`auto_conf_min=0.85` 等）、当前配置选择。
- `eval/split_frozen.csv`：28,587 行冻结划分，防止数据泄漏。
- `reports/` 下 11 份报告：`baseline_metrics.json`、`triage_metrics.json`、`rag_metrics.json`、`routing_metrics.json`、`lifecycle_metrics.json`、`skills_metrics.json`、`skills_impact.json`、`generation_metrics.json`、`closed_loop_metrics.json`、`retraining_metrics.json`、`pipeline_metrics.json`。
- 8771 分步验收台把各阶段 `build_checks()` 与实时调用拆成 140 个测试点，分环境、分诊、技能、检索、生成、路由、上线闭环、校准重训、端到端流水线、回归/冒烟共 10 组；日志按「目标 → 说明 → 方法 → 实测 → 判据 → 结论」输出，不新增报告。
- 回归测试会真实重跑五个独立页面与 `pipeline.run_all()`，并现场取报告哈希基线做隔离校验（避免把回归自身重写误判为污染）。

---

## 8. 数据隔离与安全边界

| 写入方 | 目标表 | 说明 |
| --- | --- | --- |
| 评测 / 回放 | `eval_assignments`、`route_assignments`、`triage_results`、`skill_predictions` | 冻结评测集，验收只读重算为主 |
| 一体化流水线 | `pipeline_runs`、`decision_log` | 路由固定 `dry_run` |
| 闭环门禁 | `auto_assignments`、`model_registry`、`lifecycle_config`、`feedback_events`、`drift_snapshots` | 影子 / 辅助 / 自动三档 |
| 用户工作台 | `user_requests`、`user_feedback`、`user_dispatches` | 8772 的提交、复核、评分、确认派单 |

**生产 `assignments` 表恒为 0 行**；8772 的模型在内存中训练并缓存，不写 `triage_results` / `skill_predictions`，也不改动冻结评测数据。PII 脱敏把邮箱、IP、电话、长数字替换为 `[EMAIL]` / `[IP]` / `[PHONE]` / `[NUMBER]`；`scripts/lifecycle.py` 中的 `PII_SAMPLE` 是造的演示样例，非真实个人信息。

---

## 9. 许可证与合规

| 对象 | 约束 |
| --- | --- |
| 原始工单数据集 | **`cc-by-nc-4.0`，禁止商用**；出处未记录、仓库不分发，请自行合法获取并遵守原许可 |
| 造数（工程师 / 技能 / 排班 / KB 文档） | 演示用途，无真实业务依据 |
| 项目代码 | **尚未指定开源许可证，仓库暂不含 `LICENSE`**。在作者选定并添加前，请勿假定代码可自由复用；如需开源，请自行选择 MIT / Apache-2.0 等并补上 `LICENSE` |

数据集的非商用约束**不会**因为代码换用宽松许可证而解除。整体仅供学习与内部验证，不可用于生产、真实派单或商业场景。

---

## 10. 已知限制与上传 GitHub 前检查清单

上传前建议逐项确认：

- [x] **启动脚本已配置化（已修复）**：8 个 `.bat` 统一 `cd /d "%~dp0.."`，环境检查抽到 `启动项/_env_check.bat`，不再硬编码 `F:\pgsql` / `F:\pgdata`；本机专用的 `启动项/local_env.bat` 已被 `.gitignore` 排除。配置方法见[第 4.7 节](#47-启动页面)。
- [x] **报告中的本机绝对路径已清理（已修复）**：3 份报告里的 `F:\0MyProject\...` 已改为项目相对路径（如 `eval/golden_set_human.csv`），JSON 格式已校验；日后重新生成报告时留意新产物是否又写入绝对路径。
- [ ] **大文件**：`data/raw/*.csv` 约 25 MB、`data/processed/tickets.csv` 约 30 MB，已由 `.gitignore` 排除；`eval/split_frozen.csv` 约 344 KB 保留。
      **注意 `.gitignore` 只对未跟踪文件生效**：若 `git ls-files data/` 仍列出 CSV，说明它们已被 Git 跟踪，需先移出索引（文件保留在本地磁盘）：
      ```powershell
      git rm --cached 'data/raw/*.csv' 'data/processed/*.csv'
      git commit -m "chore: stop tracking local dataset files"
      ```
      若这些 CSV 已出现在历史提交中，还要用 `git filter-repo` / BFG 清理历史，否则数据仍可被检出；最省事的做法是新建一个干净仓库重新提交。
- [ ] **本机信息自查**：`启动项/local_env.bat` 会命中本机路径，但它已被 `.gitignore` 排除，不会上传；其余命中项应逐一处理。
      ```powershell
      Select-String -Path "reports\*.json","启动项\*.bat","文档\*.md" -Pattern "F:\\0MyProject|F:\\pgsql|F:\\pgdata"
      Get-ChildItem -Recurse -File | Sort-Object Length -Descending | Select-Object -First 10 FullName,Length
      ```
- [ ] **Windows 专属**：`.bat` 与 `LOCALAPPDATA\Programs\Ollama\ollama.exe` 定位；Linux/macOS 需自行改写启动方式。
- [ ] **免密 PG**：默认依赖 trust 认证，远程/生产环境不适用。
- [ ] **无 CI、无单元测试框架**，验收依赖 8770/8771 页面与 `pipeline.run_all()`。
- [ ] **表结构已偏离最初 8 表设计**，以复盘文档为准。
- [ ] **代码许可证**：见[第 9 节](#9-许可证与合规)，需作者自行选定。

---

## 11. 发布到 GitHub

```powershell
git init
git ls-files data/            # 应为空；若列出 CSV，先按第 10 节执行 git rm --cached
git add .
git status                    # 确认 .env、__pycache__、data/ 下的 CSV 未被加入
git commit -m "Initial commit: ITSM ticket triage & resolution prototype"
git branch -M main
git remote add origin https://github.com/<your-name>/<your-repo>.git
git push -u origin main
```

`git add .` 之前请完成[第 10 节](#10-已知限制与上传-github-前检查清单)。建议把仓库设为私有，直到确认许可证与数据集合规问题都已处理。

---

## 12. 常见问题

| 现象 | 处理 |
| --- | --- |
| `psycopg2.OperationalError: Connection refused` | 先启动 PostgreSQL，`pg_isready -h 127.0.0.1 -p 5432` 应为 accepting connections |
| `CREATE EXTENSION vector` 失败 | 服务器上需先安装 pgvector 扩展本体（本项目用 0.8.0 / PG17） |
| 缺少嵌入模型 / Ollama 不可用 | `ollama pull qwen3-embedding:4b`、`ollama pull qwen3:14b`，确认 `11434` 端口 |
| 生成建议很慢 | `qwen3:14b` 首次加载约十几秒；可在 8772 关闭「生成式建议」只跑检索 |
| 端口被占用 | 关闭对应的旧服务窗口，或改 `web/*_server.py` 里的 `PORT` 常量 |
| `reports/*.json` 缺失 | 运行 `python scripts/pipeline.py` 或 8770 的 `/api/run_all` 重新生成；`data/` 缺失则先跑 `etl.py` |
| 8772 显示模型未就绪 | 后台 warmup 需要一点时间，刷新「系统状态」标签页 |
| `.bat` 提示 `[WARN] PostgreSQL is not reachable` | 复制 `启动项/local_env.example.bat` 为 `local_env.bat` 并填写 `PG_BIN` / `PG_DATA`，或手动启动 PG |

---

## 13. 延伸阅读

正式文档：

- [项目快速理解与交付沟通指南](文档/项目快速理解与交付沟通指南.md) — 一页速览、叙事线、逐段技术拆解、实测数字、高频问答与现场演示脚本（想快速讲清项目先看这篇）
- [项目实施思路](文档/项目实施思路.md) — 整体路线、取舍、常见坑与合规约束
- [数据库设计说明](文档/数据库设计说明.md) — 数据源、表结构、清洗要点与「顺序不可颠倒」的初始化步骤
- [一体化控制台使用与解读指南](文档/一体化控制台使用与解读指南.md) — 8770 页面地图、七段链路解读、接口一览、8771 分步验收台、故障排查
- [项目现状与计划实施偏差复盘](文档/项目现状与计划实施偏差复盘.md) — 现状、结构性缺口、表结构偏差

分步设计：

- [01 评测地基](思路描述/01_评测地基.md) · [02 分诊](思路描述/02_分诊.md) · [03 解决建议](思路描述/03_解决建议.md) · [04 路由到工程师](思路描述/04_路由到工程师.md) · [05 上线闭环](思路描述/05_上线闭环.md)

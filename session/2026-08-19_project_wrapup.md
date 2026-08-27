# 会话概要：项目收尾（2026-08-19）

> 本文件记录 v9 阶段节点（2026-08-19）之后的**收尾轮会话**（文档/数据/权重/配置全面整理）。
> 前置状态见 `session/2026-08-16_PROJECT_STATUS.md`（v9 暂停训练决策记录于 `index/PROJECT_V9_LIGAND_PLAN.md` §八.6）。

## 本轮做了什么

### 1. 文档全面更新（提交 `642f5ab`）
- **`WORKFLOW_GUIDE.md` 重写**为面向计算机新人的**唯一权威指南**：背景科普（生物+CS 类比）→ 整体框架（两路线 + v7/v9 双编码器）→ 数据流 → 核心模块详解 → **损失函数专章** → 参数全表 → 命令速查 → 电荷边界 → 术语表
- `README.md`、`docs/TECH|CONFIG|USAGE.md`、`CLAUDE.md` 全部同步到 v9 现状
- 修正 `run_guided.py` docstring（电荷校准现状 `enabled=false`）

### 2. 新机配置指南 + 数据组织（提交 `8b2b323`）
- 新建 `docs/SETUP_NEW_MACHINE.md`：4 类权重来源（LigandMPNN/MoMPNN clone、**v7/v9 编码器 GitHub Release**、ESMFold 自动）、conda 环境、数据重建、验证清单
- 新建 `data/README.md`（训练/验证划分防泄漏 + 重建命令）+ `data/SHA256SUMS.txt`（37 文件校验）
- 新建 `code/tests/backup_data.sh`（数据打包脚本）
- 新建 `analysis/archieved/README.md`（归档规则；旧 checkpoint 标注过时）
- 更新 `index/DOCUMENT_INDEX.md`

### 3. 权重发布与验证
- 用户网页创建 GitHub Release **`preview1.0.0`**（tag 名非计划 v1.0.0，用户确认保持现状）
- **SHA256 逐一核对**：v7 `58aca0f5...`、v9 `8ab1548f...` 远程 == 本地 ✅
- 文档下载命令统一为 `gh release download preview1.0.0`（提交 `68e7574`）

### 4. 数据与产物备份
- 数据打包：`ConfuMPNN_backup/confumpnn_data_v1_20260819.tar.gz`（2.7G）——**位于 `/data/nfs/IC` 组内 NFS 共享盘，组内新机挂载即可访问，无需额外上传**
- 产物打包：`confumpnn_artifacts_v1_20260819.tar.gz`（output/code/output/log/code/log/session，~1.1G）

### 5. 敏感信息审查
- git 仓库（当前文件 + 全部历史）：**无密码/密钥/token/真实邮箱**（提交作者用 GitHub noreply 匿名邮箱）
- 日志/产物：无真实敏感信息（唯一"命中"为蛋白序列误报 AKIA 模式）
- ⚠️ 注意：`~/.claude/settings.json` 有 DeepSeek API token（项目仓库外，不推送；本机安全性自行留意）

## 关键决策
1. **v9 暂停训练**（延续上轮）；当时视为可交付，**后经 v3 方案（2026-08-27）明确 v7/v9 为阶段性成果、非终版，v10 演进中**（见 `index/PROJECT_LOCAL.md`）
2. Release tag 用 **preview1.0.0**（用户保持现状）
3. 数据/产物备份放**组内 NFS 共享盘**（`/data/nfs/IC`），不走个人电脑中转

## 当前状态（交付）
- 新机器复现路径：`git clone` + `git clone LigandMPNN/MoMPNN` + `gh release download preview1.0.0` + 共享盘取数据 + conda 环境 → 完整复现
- 框架验证：单元测试 36/36 通过；v7 冒烟电荷命中
- **遗留**：外网电脑（非组内）需单独传输 8GB 数据；`output/` 实验历史仅在共享盘备份（git 不含）

## 产物导航（论文写作）
- 统计证据 JSON：`output/*_stats.json`（ph_scan / transfer / generalization_v9）
- 报告：`analysis/report/`（E1 → v9 泛化验证完整链）
- 完整产物：`ConfuMPNN_backup/confumpnn_artifacts_v1_20260819.tar.gz`

# 项目迁移 & Git 归档策略（2026-09-04 定稿）

> 目的：本仓库后续要整体迁移到其他电脑/服务器复现；本文件是**归档/提交的唯一规则**，与实验科学进展文档（`session/`、`analysis/report/`）分开，请勿在此记录实验结论。

## 一、什么进 git（本仓库，跟随迁移）
1. **代码**：`code/`、脚本、配置文件（`index/`、`docs/`）。
2. **报告/记录**：`analysis/report/`、`session/` 会话与检查点日志、`figure/`。
3. **日志**：`log/` 与 `code/log/` 全量（训练/验证/诊断 stdout），便于多机对照与论文复现（`.gitignore` 已不再忽略）。
4. **关键小数据（强制入库）**：
   - 验证集 **manifest JSON**：`data/validation_pdbs/*.json`（`.gitignore` 已放行）；
   - **output 顶层结果 JSON**：诊断/校准/组成/统计等论文关键数字（`!output/*.json` 已放行，随各验证块自动入库）；
   - 各 `tm_sol_*/tm_sol_summary.json` 等**子目录汇总 JSON**：由检查点/终局 `git add -f` 显式加入。

## 二、什么走 GitHub Release（不入 git）
- **自训权重 / 条件编码器 `*.pt`**：每个版本只传**确认后的最终件**（如 `finetune_ligand_v14_rna/finetune_epoch050.pt` + `condition_encoder_last.pt`），命名含版本号与说明。确认“更新好了/不再重训”后上传，勿频繁覆盖。
- 骨架权重（LigandMPNN `model_params/`、MoMPNN）为上游可下载，**不传**。

## 三、什么走网盘 / NAS（项目结束后人工上传，非 git）
- `data/` 大文件：CATH(`data/cath`)、配体训练集(`data/ligand_train` 的 labels.npz + all_pdb)、`data/validation_pdbs/*.pdb`、拆链/候选池等（校验用仓库内 `data/SHA256SUMS.txt`、`data/README.md`）。
- `output/` 重型中间产物：`generalization_*/`（采样序列 + ESMFold 折叠结构）、`finetune_*/epoch*.pt`、`tm_sol_*/`、预解析缓存等。
- 上传后在本文件下方“网盘备份记录”登记日期与位置。

## 四、长任务（训练/验证链）的归档节奏
- **只在“每一块工作（阶段）完成”后**做一次：总结该阶段数据 → 更新对应 `session/` 自动日志 → `git add`（本次阶段新增的 `*_clean` 顶层 JSON + log）→ commit → `git push`。
- 阶段进行中**不**反复 push（避免 commit 噪音）。
- 异常终止 / 到关键决策点：**停下来问用户**，不自动续跑或下结论。

## 五、红线
- 不把 **uid/账号权限等运维琐事**写进实验记录（那是另一话题，按需单独记）。
- 不混提交：一个 commit 只对应一次“归档动作 + 其结果”，不要把多年积压一股脑塞进一个 commit。
- 大文件绝不 `git add -A` 硬塞（GitHub 体积限制）。

---
### 网盘备份记录
| 日期 | 内容 | 位置/链接 |
|---|---|---|
| （待项目收尾填写） | data/ 大文件 + output 重型产物 | |

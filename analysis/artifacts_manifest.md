# ConfuMPNN 实验产物清单（论文写作导航）

> 本文件索引项目**全部实验产物**的位置、内容与是否进 git。
> 更新时间：2026-08-19（v9 节点收尾；v10 演进中，产物导航仍有效）。
> 用途：写论文时快速定位证据；区分「git 内的文档证据」与「备份包内的原始产物」。

---

## 一、产物总体分布

| 位置 | 内容 | 体积 | 是否进 git |
|------|------|------|-----------|
| `analysis/report/` | **实验报告**（E1 → v9 泛化验证，23 份）| 小 | ✅ 在 git |
| `analysis/evidence/` | **关键统计 JSON**（论文直接引用的数值证据）| 小 | ✅ 在 git |
| `output/` | **v7/v9 最终产物**（checkpoint / 验证结果 / 统计）| 592M | ❌ 备份包 |
| `code/output/` | **早期产物**（Phase 1-6 验证 / finetune_vN / phase3_*）| 472M | ❌ 备份包 |
| `log/` + `code/log/` | 训练/实验日志 | 6M | ❌ 备份包 |
| `session/` | Claude Code 会话概要 | 84K | ✅ 在 git |
| `data/` | 训练/验证数据（见 `data/README.md`）| 8G | ❌ 共享盘 |

**完整备份包**（含 output/code/output/log/code/log/session，~1.1G）：
`ConfuMPNN_backup/confumpnn_artifacts_v1_20260819.tar.gz`（SHA256 见同目录 `.sha256`）

---

## 二、`analysis/evidence/` —— 论文可直接引用的统计证据

| 文件 | 内容 | 对应报告 |
|------|------|---------|
| `generalization_v9_stats.json` | v9 泛化验证汇总（10 蛋白 × 5 臂 × H1/H2）| `2026-08-19_v9_generalization_validation.md` |
| `ph_scan_stats.json` | 多 pH 温和区复现天然（3 蛋白 × 3 pH × n50）| `2026-08-18_model_validation_phscan.md` |
| `ph_scan_sanity.json` | 序列合理性（无 X / 核心疏水 / 折叠）| `2026-08-18_seq_sanity_and_transfer.md` |
| `transfer_stats.json` | 迁移应用能力（5 新蛋白）| `2026-08-18_seq_sanity_and_transfer.md` |
| `transfer_v9_stats.json` | v9 配体模式迁移（1MBN/4DFR/1FQG）| `2026-08-18_v9_ligand_training.md` |

## 三、`output/` 关键目录（备份包内）

| 目录 | 内容 | 论文用途 |
|------|------|---------|
| `output/finetune_v7/` | v7 编码器 30 epoch checkpoint + 最终权重 | 训练曲线（log/train 日志）|
| `output/finetune_ligand_v9/` | v9 编码器 30 epoch checkpoint + 最终权重 | 训练曲线 |
| `output/generalization_v9/` | 80 臂 × 30 序列 seqs.fa + 回折 PDB + tm | 泛化验证原始数据 |
| `output/transfer_v9/` | 迁移验证序列 + 打分 | 迁移结果原始数据 |
| `output/ph_scan*/` | pH 扫描序列 + 回折 | pH 响应原始数据 |
| `output/finetune_v7_validate/` | v7 复验 6 臂 × n20 | v7 验证原始数据 |

## 四、`code/output/`（早期产物，备份包内）

- `finetune_v1~v6*`：各轮训练 checkpoint + 验证（v2-v6 为**过时版本**，最终版为 `output/finetune_v7`）
- `phase3_*`：Phase 3 条件注入验证（pH 响应 / 防失控 / 电荷校准）
- `e1_*`：E1 三目标对照（MoMPNN vs LigandMPNN）
- `guided_*`：示例蛋白引导采样输出

## 五、日志（备份包内）

| 日志 | 内容 |
|------|------|
| `log/v9_train.log` | v9 训练全日志（30 epoch 损失曲线）|
| `code/log/train*.log` | v7 及早期训练日志 |
| `code/log/compare_*.log` | E1 对照采样日志 |
| `code/log/*.log` | 各实验脚本运行日志 |

## 六、外部工具（gitignore，获取方式）

| 工具 | 用途 | 获取方式 |
|------|------|---------|
| `LigandMPNN/` | 逆折叠模型源码 + 权重 | ✅ `git clone --recursive https://github.com/dauparas/LigandMPNN.git` |
| `MoMPNN/` | 默认生成器权重 | ✅ `git clone https://github.com/Qivon7/MoMPNN.git` |
| `protein_sol_mcp/` | 可溶打分 %sol | ✅ `git clone https://github.com/MacromNex/protein_sol_mcp.git` |
| `TemBERTure/` | 热稳打分 Tm | ⚠️ **无 git，备份包**：`confumpnn_tools_temberture_v1_20260819.tar.gz` |
| `foundry/` | RosettaCommons 工具库（备选）| ✅ `git clone https://github.com/RosettaCommons/foundry.git` |

## 七、论文写作建议引用路径

1. **结论数值**：优先引 `analysis/evidence/*.json`（git 内，可复现）
2. **方法细节**：`WORKFLOW_GUIDE.md` + `docs/TECH.md`
3. **实验叙述**：`analysis/report/` 对应报告
4. **原始数据**：从备份包解压 `output/` 对应目录

> ⚠️ 备份包内文件含机器特定路径（`/data/nfs/IC/baokun_yu/...`），引用时需替换为通用路径说明。

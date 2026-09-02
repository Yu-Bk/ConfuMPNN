> ⚠️ **再更新（2026-09-02 晚）**：按用户策略大规模扩充核酸数据后 v14 第 3 次重启。训练集 **labels_v14_final.npz（5371 域，RNA/DNA 414=7.7%：DNA 155 + 非核糖体RNA 108 + 核糖体RNA 148）**。验证集 11 蛋白 held-out 全 leak=False。详见 `session/2026-09-02_v14_rna_data_a1_global.md` §9。

> ⚠️ **更新（2026-09-02 下午）**：首次 v14 训练（labels_v14_merged, 5148 域）epoch1/2 已作废。用户重构数据/验证集后重启：训练集 **labels_v14_final.npz（5166 域）**，验证集 11 蛋白（validation_manifest_v14_final.json），详见 `session/2026-09-02_v14_rna_data_a1_global.md` §8。

# v14 配体重启：RNA/DNA 数据扩充 + A1 全局化报告（2026-09-02）

> **执行**：配体线 executor（session/2026-09-02_v14_rna_data_a1_global.md）
> **训练**：`output/finetune_ligand_v14_rna/`（GPU4，50ep，运行中）

## 一、结论摘要

1. **数据扩充完成**：加入 191 个 RNA/DNA 结合蛋白唯一域（核糖体蛋白为主 + 一般核酸结合复合物），
   合并后训练集 5148 域（旧 4957 + 191），all_pdb 加 191 symlink。
2. **number_of_ligand_atoms 16→25 全脚本修正**（对齐权重 atom_context_num=25）。
3. **A1 全局化实现**（`--pocket_mode global`）：计数锚从 pocket 扩到 surface∪pocket，floor 0.8/ceil 1.3/λ 0.3。
4. **dry-run 通过**（50 域混合 0 NaN / 0 分区失败）。
5. **v14 训练已启动**（GPU4，50 epochs，理由：RNA 新类型 + 25 原子首次 + v13 30ep 未收敛）。

## 二、数据收集清单

### 拆链产物（260 个单链样本）
| 来源类型 | PDB | 蛋白链数(拆出) | 说明 |
|---------|-----|--------------|------|
| 核糖体主源 | 4V4T | 46 | T. thermophilus 70S |
| | 9RVC | 44 | 高分辨 70S |
| | 4YBB | 95 | 70S（约 2 拷贝，序列去重后少） |
| 补充 | 5AVC | 8 | 人核小体组蛋白+DNA |
| | 5GIN | 9 | box C/D RNP + guide/substrate RNA |
| | 6IFL / 9ASH | 7/9 | 型 III CRISPR-Csm + RNA |
| | 9FB4 | 6 | SV40 large T + DNA |
| | 5VVL | 6 | Cas1-Cas2 + DNA |
| | 7OUH | 4 | STLV intasome + DNA |
| | 4NOD | 4 | TFAM + DNA |
| | 3WVK | 4 | HindIII + DNA |
| | 2V3C | 4 | SRP54-SRP19 + 7S RNA |
| | 1BP7 | 4 | I-CreI meganuclease + DNA |
| | 7V9X | 3 | retron-Ec86 effector + RNA/DNA |
| | 3HOT / 3ADB / 2ZZN / 8ZDR | 2/2/2/1 | Mos1 / tRNA激酶 / aTrm5-tRNA / Cas9d |

### 配体保留决策
- 截断 **15Å**（蛋白链任重原子）：对核糖体蛋白与 rRNA 界面保留完整局部 RNA 片段（100-1400 配体原子），
  避免过长无关 RNA；对 LigandMPNN 每残基取最近 25 原子特征化，15Å 内原子数不撑爆。
- 去水、去其他蛋白链（保证单蛋白链）；保留 RNA/DNA/小分子/配位离子（金属与配体配位可保留）。

### 标签质量
- 每域 8 pH（uniform 4-10）净电荷/pI，序列去重（同源核糖体蛋白跨结构、组蛋白多拷贝）
- 排除与旧训练集序列重复；每个拆链 QC（parse_PDB L 匹配、mask≥0.9、Y>0）
- 191 域长度 50-446，charge@7.4 mean +8.7（碱性，核酸结合特征）

## 三、A1 全局化代码 diff 摘要

- `code/src/v12_losses.py::pocket_count_loss`：参数 `normalize`（分数化，除以 native 计数）
  + `min_abs_cap`（N=0 死锁保护，默认 2）。
- `code/train_finetune.py`：
  - `--pocket_mode` 增 `global`；keep/global 都算三块互斥分区（core/pocket/surface）
  - global 计数区 = charge_surf_mask（surface∪pocket），A2 extra_mask 同样扩展
  - global 传 `normalize=True`
- 保留原 pocket（keep）模式代码与语义（v13 复现）。

### dry-run（log/v14_dryrun.log）
- 50 域（20 旧小分子 + 30 RNA）：0 NaN、0 分区失败、checkpoint 写成功。
- RNA 蛋白三块分区示例：4V4T_AB pocket=37/234 core=100 surface=116；2ZZN_A pocket=116/336 core=113 surface=153。
- epoch1 total=10.26 ce=1.69 charge=11.93（首 epoch 未收敛属正常）。

## 四、训练启动确认

| 项 | 值 |
|----|----|
| 命令 log | `log/v14_ligand_train.log`（stdout: `log/v14_ligand_train_stdout.log`） |
| out_dir | `output/finetune_ligand_v14_rna/` |
| GPU | cuda:4（启动时 6GB/99% 占用，进程存活） |
| epochs | 50（v13 30ep 未收敛 + RNA 新类型 + 25 原子首次） |
| 数据 | 5148 域 × 8 = 41184 样本（旧 4957 + RNA 191） |
| 超参 | global floor0.8 / ceil1.3 / λ_pocket0.3 / cutoff8；λ_target0.2；v12 frac0.5/gravy0.4/λ0.2 |

## 五、验证集（v14 配体模式）

见 `data/validation_pdbs/validation_manifest_v14_ligand.json`：
- 保留单体：1AZM/1AS2/2FEO/5CQH/1CGE/1A65/1BJ4（小分子/核苷酸/长）
- 删除二聚体：1C6O/1AXW/1AG0
- 新增 held-out 核酸结合（无泄漏）：21KL_A/3MXB_A/4GDF_A/5ZR1_B/8DR1_A/9DWG_L

## 六、下一步（训练完成后）

1. 组成分析（带电总数 0.7-1.3×，重点看删减是否根治，含 RNA 蛋白）
2. 配体诊断 slope（校准后 valid ∈[0.9,1.15]）
3. 泛化 n50（新 manifest 13 蛋白 × 5 臂）→ H2/H1/H4
4. H3 聚集 + Tm/Sol 复测（对照 v12.2 配体 9/50、v13 17/50）

产物与日志全部 git 归档（代码、data 清单、session/report；训练输出 gitignore）。

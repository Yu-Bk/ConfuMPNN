# v12.3 蛋白模式扩长蛋白重训 + 验证集重构（2026-09-02 执行）

## 目标
v12.2 蛋白模式（MoMPNN）为当前最优（slope 1.00 / H2 72% / Tm-Sol 0/50），短板 = **长蛋白训练量不足**
（CATH 6710 训练域 L>400 仅 132 = 2%；验证 1BJ4/470、1A65/504 为长度 OOD，H2 失败）。
v12.3 = v12.2 基础上**补入 S40 长蛋白域重训**（不改 v12.2 超参，单一变量 = 加长蛋白数据）。

---

## A. 可行性评估

### A1. S40 长蛋白统计（`/tmp/cath_s40_stats.json`，解析 34653 个 dompdb 文件实测）
| 长度 | 域数 | CATH class 1/2/3 分布 |
|---|---|---|
| L>300 | 2488 | 597 / 436 / 1455 |
| **L>400** | **617** | 229 / 109 / 279 |
| L>450 | 311 | 124 / 57 / 130 |
| L>500 | 154 | 59 / 32 / 63 |
| L>600 | 44 | — |
| L>700 | 25 | — |
| max | **1202**（1y5lA02） | |

- S40 域**全部单链**（多链=0，天然规避同源二聚体）；含 HETATM 6746 个（多为水/离子，protein parse 忽略）；非标残基域仅 37 个（候选长域命中 4 个，已剔除）。
- **结论：长蛋白池充足**，可支撑训练补长。

### A2. 采样方案（保留原 6710 + 追加，最小改动）
- **v12.3 训练 = 原 v12.2 的 6710 域 + S40 中 L≥400 且未入 v12.2 train(6710)/holdout(1176) 的域**。
- 候选粗筛（CATH fasta L≥395，排除已用/holdout/验证 PDB code/4 个非标域）568 → parse 后 L≥400 **455 个**。
- 追加后：**7165 域**，L>400 = 581（**8.1%**，v12.2 为 2%），L>450 = 287，L>500 = 144，max 1202。
  - 硬上限 ~9.5%（S40 未用 L≥400 池仅 ~534），8.1% 是保留 6710+全收长候选的实际可达值，略低于建议 10% 但在论证范围（较 v12.2 提升 4 倍，覆盖 1A65/504、1BJ4/470 OOD 区 + 外推至 1200）。
- CATH class 比例保持：v12.2 1/2/3 = 26.5%/21.4%/52.1% → v12.3 = 26.0%/20.4%/49.9%（微偏 αβ 少，可接受）。
- 标签与 v12.2 完全同构：domain_ids/seqs/coords(Cα)/pH/charge/pI，每域 8 pH Uniform[4,10]。
- **不动 condition_defaults.yaml**（μ/σ 沿用 v12.2，保持推理归一化一致）。

### A3. 长蛋白引入风险排查结论
1. **SASA/frac_sasa 预解析**：`src/sasa.py` 用 freesasa+Bio.PDB，按残基线性；455 长域 SASA 一次性 ~10min，内存线性，无异常（v12.2 已有 L=950 域成功）。
2. **GPU 显存/batch**：train_finetune 为**逐域 step**（非 batch padding），单 step 只处理 1 域 × 8 pH → 峰值由最长域决定。**v12.2 已成功训练含 950 残基域**；dry-run 实测最长 1202 域 1-epoch 无 OOM（缓存 0.79GB/50 长域）。
3. **parse_PDB/特征化**：新增 455 域全部经 train_finetune 同款 parse_PDB+featurize 预检 **ok=455/fail=0**（关键：训练中 skip=0 → 无 pH/charge 段错位）。
4. **非蛋白残基**：S40 域基本纯蛋白；37 个含非标 CA 的域中 4 个落在长候选，已剔除。
5. **多链**：S40 域单链天然保证。

---

## B. 数据前处理
- 脚本：`code/tests/build_v12_3_augment.py`（新写，复用 build_labels.py 同款 parse_domain/net_charge/find_pI）。
- 产物：`data/cath/labels_v12_3_train.npz`（7165 域 × 8 pH = 57320 样本）。
- 校验：域数/长度直方图/L>400 占比/class 比例/pH range[4,10] 全部打印通过；新增域 parse 预检 455/455。
- 泄漏保护：候选排除 10 个现有验证蛋白 PDB code + E 新增 5 个（1cdg/13bb/1acc/1ayl/1bpm）+ holdout 域。

## C. 训练前检查 + dry-run
- 环境：GPU6（显存空 1.3GB/140GB，但有人小进程占 util，训练仍正常）；confumpnn python。
- dry-run：50 长域（含最长 1202）临时 labels → 1-epoch **0 NaN、0 崩溃**，checkpoint 正常（`output/finetune_v12_3_dryrun`）。
- v12.2 训练命令已从 `log/v12_2_train_mompnn.log` + `session/2026-08-29_v11_ablation.md §9.2` 精确重建并沿用。

## D. 训练启动（PID 1400836，GPU6）
```bash
PYTHONPATH=code nohup setsid ~/miniconda3/envs/confumpnn/bin/python code/train_finetune.py \
  --device cuda:6 --epochs 30 \
  --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt \
  --labels data/cath/labels_v12_3_train.npz --dompdb data/cath/S40/dompdb \
  --lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 \
  --charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15 \
  --decouple_perturb --decouple_range 12.0 \
  --ph_aware_filter --structure_boost 1.5 \
  --v12_supervision --frac_floor 0.5 --gravy_margin 0.4 --lambda_v12 0.2 --lambda_target 0.2 --sasa_threshold 0.25 \
  --out_dir output/finetune_v12_3 \
  --log_file log/v12_3_train_mompnn.log --log_progress log/v12_3_train_mompnn_prog.json \
  > log/v12_3_train_mompnn.stdout 2>&1 &
```
- 回退保障：v12.2 权重 `output/finetune_v12_2/finetune_epoch030.pt` + `condition_encoder_last.pt` 已 cp 到 `/tmp/v12_2_backup/`（md5 一致），未改动原目录。
- 启动确认：进程存活、日志正常（7165 域确认）、GPU6 显存开始累积、无 NaN。
- 预计 ~30ep × ~12-14min + 预解析 ~20-30min ≈ **7-9h**（主 session cron 监控）。

## E. 验证集重构（不阻塞训练）
- `data/validation_pdbs/validation_manifest_v12_3.json`（新写）：
  - **删除同源二聚体**：1C6O/1AXW/1AG0（native 回折 TM~0.5 为 ESMFold 工具限制，H1 无法单链评估）。
  - **保留 7 单体**：1AZM(258)/1AS2(315)/2FEO(221)/5CQH(186)/1CGE(162)/1A65(504)/1BJ4(470)——1A65/1BJ4 保留作 v12.3 长蛋白改进检验。
  - **新增 2 未训练单链长蛋白**：13BB(560, 乙酰乳酸合酶, TPP)、1CDG(686, 环糊精糖基转移酶)——扩展长蛋白 OOD 长度覆盖（>504）。
  - 9 个蛋白全部单链（链 A）、不在 v12.3 训练集（S40 6710+455 域）、leak=False。
- 兼容性：validate_generalization.py 以 `items[].pdb/path` 读取，额外字段 L/note 经 `.get()` 访问不影响。

## 后续（主 session）
1. **训练监控**：cron/手动看 `log/v12_3_train_mompnn.log`（epoch 30/30、无 NaN、checkpoint 完整 `output/finetune_v12_3/`）。
2. **验证链**（v12.3 checkpoint 替换）：
   - 17 蛋白响应诊断（valid 区 slope ∈ [0.9,1.15]）→ 建/复用校准表（新表应重拟合，因分布变长）。
   - 组成分析（D/K vs native，防删减/过度添加）——**长蛋白组成是重点**。
   - 泛化验证：`run_v12_1_validation.sh` 换 `--manifest validation_manifest_v12_3.json` + v12.3 checkpoint（9 蛋白 × 5 臂 × n30）。
   - H1 折叠（ESMFold TM，长蛋白尤其看 1A65/1BJ4/13BB/1CDG）、H2 电荷、H4 PROPKA、Tm/Sol。
3. **对比口径**：v12.3 vs v12.2 长蛋白（1A65/1BJ4）H2 是否改进 = 本轮核心结论。

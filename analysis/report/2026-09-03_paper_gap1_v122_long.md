# paper gap1：v12.2 4 长蛋白两口径泛化基线（补测）

日期：2026-09-03
任务：补测 **v12.2 蛋白模式权重**对 4 个长蛋白（1A65/1BJ4/13BB/1CDG）的两口径泛化 H2，
形成 v12.3 同口径对照的 v12.2 侧基线，支撑论文叙事 "v12.2 在长蛋白/深负蛋白上不行 →
v12.3 补长蛋白数据后**一定程度**缓解"。

## 1. 方法与产物

- 模型：v12.2 condition encoder `output/finetune_v12_2/finetune_epoch030.pt`（epoch 30）
  + MoMPNN backbone `MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt`，
  protein 模式（无配体）。
- Manifest（新写，只含 4 长蛋白，条目对齐 v12.3 manifest）：
  `data/validation_pdbs/validation_manifest_v12_2_long.json`
- 臂定义：native/n2/p2/n8/p8（native = round(native_charge@7.4)；Δ = 0/−2/+2/−8/+8），每臂 n=30，pH 7.4，
  seed_base=2000。H2 = |mean_charge − target| ≤ 2.0。
- 口径 A（小样本现场标定）：`code/tests/build_calibration_small.py`（4 蛋白 × 5 target
  native±[8,4,0,4,8] × n_per=10 = 50 条/蛋白，LOOCV 稳定性校验，均 <3 → 全部 reliable 用自身 slope）
  → 新表 `output/paper_gap1_v122_long/charge_calibration_v12_2_long_small.json`；
  随后 `validate_generalization.py --calibrate auto` resample。
- 口径 B（big-global，纯训练域只读表）：`output/charge_calibration_v12_2_big.json`
  （global slope 1.4587 / intercept −0.7518，92 训练域），`--calibrate global` resample。
- 小样本标定结果（slope/int/LOOCV，全可靠）：
  - 1A65: slope 1.604, int 1.70, LOOCV 2.74
  - 1BJ4: slope 3.007, int 3.31, LOOCV 0.82
  - 13BB: slope 1.587, int −2.53, LOOCV 2.35
  - 1CDG: slope 2.470, int −2.91, LOOCV 2.73
- 产物目录（新目录，未动原校准表）：
  `output/paper_gap1_v122_long/generalization_small/protein/<pdb>/validation.json`
  `output/paper_gap1_v122_long/generalization_bigglobal/protein/<pdb>/validation.json`
  汇总：`output/paper_gap1_v122_long/summary_h2_table.txt`
  日志：`log/v12_2_long_2caliber.stdout`、`log/v12_2_long_{calib_small,small_sample,bigglobal_sample}.log`
  驱动脚本：`code/tests/ligand_v9/run_v12_2_long_2caliber.sh`
- GPU：cuda:3；未发生中断，一次性跑完。

## 2. 逐蛋白两口径 H2（v12.2，本次实测）

判据 H2：dev ≤ 2.0。每臂 dev = |mean_charge − target|（n=30 生成序列；native 参考仅写入 seqs.fa，
不进 mean/dev —— validation.json 内 dev 字段已排除 native，`n_generated: 30`）。

### 口径 A：小样本现场标定

| pdb  | native_q@7.4 | native | n2 | p2 | n8 | p8 | H2 | v12.3 同口径 |
|------|--------------|--------|----|----|----|----|----|----|
| 1A65 | −26.85 (L504) | 3.51 | 3.91 | 3.75 | 2.91 | 4.07 | **0/5** | 2/5 |
| 1BJ4 | +0.42 (L470)  | 1.83* | 1.73* | 1.65* | 1.81* | 1.40* | **5/5** | 5/5 |
| 13BB | −12.80 (L552) | 2.99 | 2.62 | 2.67 | 3.20 | 3.33 | **0/5** | 1/5 |
| 1CDG | −8.94 (L686)  | 0.32* | 0.28* | 0.43* | 0.06* | 1.42* | **5/5** | 4/5 |

v12.2 合计 **10/20 = 50%**；v12.3 合计 12/20 = 60%（* = 该臂命中）。

### 口径 B：big-global（未标定，纯训练域表）

| pdb  | native_q@7.4 | native | n2 | p2 | n8 | p8 | H2 | v12.3 同口径 |
|------|--------------|--------|----|----|----|----|----|----|
| 1A65 | −26.85 (L504) | 4.01 | 4.20 | 3.96 | 4.52 | 3.24 | **0/5** | 1/5 |
| 1BJ4 | +0.42 (L470)  | 3.44 | 0.74* | 5.37 | 4.17 | 10.99 | **1/5** | 1/5 |
| 13BB | −12.80 (L552) | 0.44* | 0.53* | 0.20* | 0.54* | 1.03* | **5/5** | 4/5 |
| 1CDG | −8.94 (L686)  | 7.78 | 9.53 | 6.38 | 14.09 | 3.51 | **0/5** | 3/5 |

v12.2 合计 **6/20 = 30%**；v12.3 合计 9/20 = 45%。

## 3. v12.2 vs v12.3 同口径对照

| 口径 | pdb  | v12.2 H2 | v12.3 H2 | Δ | 说明 |
|------|------|----------|----------|----|------|
| small | 1A65 | 0/5 | 2/5 | +2 | 深负长尾改进 |
| small | 1BJ4 | 5/5 | 5/5 | 0 | 持平 |
| small | 13BB | 0/5 | 1/5 | +1 | 轻微改进（仅 p8 命中） |
| small | 1CDG | 5/5 | 4/5 | −1 | 轻微回退（p8 2.97 差一臂） |
| bigglobal | 1A65 | 0/5 | 1/5 | +1 | 深负长尾改进 |
| bigglobal | 1BJ4 | 1/5 | 1/5 | 0 | 持平 |
| bigglobal | 13BB | 5/5 | 4/5 | −1 | 轻微回退（p2 2.12） |
| bigglobal | 1CDG | 0/5 | 3/5 | +3 | 超长外推大幅改进 |

（v12.3 数值为任务提供权威值，且已从 `output/generalization_v12_3_{calib_small,bigglobal}/protein/<pdb>/validation.json`
逐一核对：1A65 small 2/5、1BJ4 5/5、13BB 1/5、1CDG 4/5；big-global 1A65 1/5、1BJ4 1/5、13BB 4/5、1CDG 3/5。）

## 4. 结论（供论文叙事使用）

**v12.2 失败基线（本次补测确认）**
1. **1A65（深负 native −26.9、L504，v12.2 OOD）**：两口径**全失败**（small 0/5、big-global 0/5），
   所有臂 dev 均 2.9–4.5——v12.2 对深负长尾长蛋白电荷控制系统性欠冲/过冲，小样本现场标定也救不回
   （这是 v12.2 时代已知的 1A65 0/5）。这是叙事最硬的"v12.2 不行"证据。
2. **1CDG（超长 L686、mild 负 −8.9，v12.2 OOD）**：big-global **严重失败 0/5**（native dev 7.78、
   n8 dev 14.09——超长外推下全局响应偏负严重），但**小样本现场标定可救到 5/5**（自身 50 条拟合的
   slope 2.47 补偿了该蛋白的响应）。说明 1CDG 的失败是"无标定/全局外推"型失败，可被现场标定覆盖。
3. **13BB（L552、native −12.8，v12.2 未见）**：两口径分裂——small **0/5**（自身小样本 slope 1.59/int −2.53
   校正后仍全臂 dev>2，标定反而带偏）、big-global **5/5**（落在全局线性区，控制极准）。
   → "13BB 新增样本 v12.2 必失败"不成立，仅 small 口径失败。

**v12.3 补数据后"一定程度"缓解（对照支持）**
- 最稳收益集中在 **OOD 深负/超长**：1A65 两口径 0→2、0→1；1CDG big-global 0→3。
- 1BJ4 持平（small 5/5、big-global 1/5），本就不是 v12.2 的长蛋白痛点。
- 代价/边界：1CDG small 5→4、13BB big-global 5→4 轻微回退；13BB small 仅 0→1。
- 净变化：small 10/20→12/20（+2，1A65 +2 为主）；big-global 6/20→9/20（+3，1A65 +1、1CDG +3 为主）。
- **措辞必须"一定程度"**：1A65 即便 v12.3 也只是 2/5（small）/1/5（big-global），远未根治；
  v12.3 的收益不是全面超越，而是把 OOD 长蛋白从"完全失败"拉到"部分可控"，并在个别臂/口径有回退。

## 5. 复现性与方法学备注
- 小样本标定结果与 v12.2 历史小样本表完全一致（1A65 slope 1.604/int 1.70、1BJ4 3.007/3.31），
  且 small resample 的 1A65 dev（3.51/3.91/3.75/2.91/4.07）、1BJ4 dev（1.83/1.73/1.65/1.81/1.40）
  与 v12.2 历史 small 批逐位相同 → 确定性复现成立。
- 本次 4 蛋白均 LOOCV<3（reliable），无 unreliable 回退 global 的分支发生（与 v12.3 侧 1A65 unreliable 不同）。
- H2 统计未把 seqs.fa 末尾 native 参考计入（validation.json dev 仅由 30 条生成序列算出）。

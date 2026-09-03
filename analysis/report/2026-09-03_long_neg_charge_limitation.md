# 归档：长 × 大电荷变化（深负/深正）可设计性有限 — 两模式证据汇总（2026-09-03）

> **触发**：用户对 v14 配体验证集的方法论修正——2E9R_X（FMDV RdRp, L476, native −10.1）
> 被判定不具 RNA"结合蛋白"代表性（核糖体 RNA 结合蛋白天然正电；2E9R_X 为长负聚合酶，
> 验证系统性过冲），归入与蛋白模式 1A65 同类的"**长 × 大电荷变化可设计性有限**"档，
> 从标准验证集移除归档，由 5O60_E（核糖体蛋白 E, L209, native +11.18）替换作 RNA 代表。
> 本报告归档该档证据与 2E9R_X 数据路径，供论文"能力边界"章节引用。
> 主报告：`analysis/report/2026-09-03_validation_standards.md` §4.2（in 已置换）。

---

## 一、核心结论

模型在**长蛋白 × 较大电荷变化**（尤其深负工程化）上**可设计性有限**，且跨蛋白/配体两种模式一致：

- **长蛋白温和调整（native ± 2）可靠**：如蛋白模式 1BJ4（L470, native≈0）v12.2/v12.3 小样本均 5/5。
- **长蛋白中度调整（native ± 8）部分可靠**：部分蛋白 n8/p8 命中、部分过冲/欠冲。
- **深负长蛋白工程化超出当前能力**：native ≤ −10 档的长蛋白即使小样本现场标定也无法救回
  （1A65 深负欠冲、2E9R_X 负向过冲；标定只能校正推理侧增益，救不了"模型不会生成"的饱和区）。

---

## 二、证据① 蛋白模式 1A65（L504, native −26.9，分布外深负）

| 口径 | v12.2 | v12.3 | 说明 |
|---|---|---|---|
| 小样本现场标定 | **0/5** | **2/5** | 补负电富集长蛋白数据后仅"**一定程度缓解**"，未根治 |
| big-global（纯训练域） | — | **1/5** | 开箱基本不可设计 |

- native_q −26.85 落在 v12.3 训练 q 分布 ~2.8% 分位 + L504 超训练 max500 → 覆盖边界预期。
- 机制：达到深负需插入大量 D/E，对长蛋白显著牺牲疏水核心/折叠 → 生成侧"物理阻力"饱和 → 线性校准在弯曲段失效。
- 数据源：`analysis/report/2026-09-03_v12_3_vs_v12_2_final.md`、`2026-09-03_validation_standards.md` §4.1/4.3；
  `output/generalization_v12_3_calib_small/`（1A65）、`output/generalization_v12_2*/`（v12.2 0/5）。

## 三、证据② 配体模式 2E9R_X（L476, native −10.1，FMDV RdRp）——本报告归档主体

### 3.1 为什么归档
- 它代表不了"RNA 结合蛋白"：核糖体 RNA 结合蛋白（如 5O60_E/21KL_A）**天然正电**；2E9R_X 是**长负 RNA 聚合酶**，
  其失败根因是"长 × 负向大电荷变化"，而非 RNA 结合本身。留在标准集会让 RNA 类别成绩被一个非典型成员污染。

### 3.2 响应与校准诊断（`log/v14_small_cal.log`）
小样本现场标定（5 target × n10）：slope **2.337** / intercept −4.41，LOOCV 3.07 → **unreliable**。
raw 响应（target → 生成电荷）：−18→**−44.2**、−14→**−39.6**、−10→**−29.7**、−6→−16.1、−2→−9.3。
→ 模型对负 target 的系统性**过冲**极强（响应斜率远大于 1），属生成侧行为，非简单增益偏移。

### 3.3 三口径采样 H2（n30，per-arm dev 判据 ≤2.0）
| 口径 | H2 | 各臂 mean（target→mean, dev） |
|---|---|---|
| big-global（`generalization_ligand_v14_big`） | **0/5** | native(−10→−18.09, 8.09) / n2(−12→−21.31, 9.31) / p2(−8→−15.19, 7.19) / n8(−18→−30.36, 12.36) / p8(−2→−7.33, 5.33) |
| 小样本现场标定（`generalization_ligand_v14_small`） | **0/5** | 同 big（标定 unreliable → 自动回退 global → 结果同 big） |
| per-protein 表内（诊断网格，`generalization_ligand_v14`） | **2/5** | native(−10→−7.66, 2.34) / n2(−12→−9.77, 2.23) / p2(−8→−6.07, 1.93✓) / n8(−18→−15.16, 2.84) / p8(−2→−1.16, 0.84✓) |

- 关键：即便用 per-protein 大 slope 2.34 校准，**负向臂（native/n2/n8）仍全部过冲超阈**，
  只有靠正的方向靠拢的 p2/p8 命中 → **标定救不了负向深调，属生成侧上限**（与 1A65 欠冲方向相反但同档）。

### 3.4 归档数据路径（保留，不删除；统计用新 manifest 自动排除）
- `output/generalization_ligand_v14/ligand/2E9R_X/`（per-protein 口径，2/5）
- `output/generalization_ligand_v14_big/ligand/2E9R_X/`（big-global 口径，0/5 过冲）
- `output/generalization_ligand_v14_small/ligand/2E9R_X/`（小样本口径，0/5 过冲）
- `output/charge_calibration_v14_small.json` → `per_protein.2E9R_X`（slope 2.337, unreliable；v2 表保留该条目但标注已归档）
- `output/v14_ligand_gen_stats{,_big,_small}.json` 中的 2E9R_X 行
- manifest 归档注：`data/validation_pdbs/validation_manifest_v14_final.json` / `_v14_in.json` → `archived` + `replaced["2E9R_X"]`

---

## 四、替换：5O60_E（核糖体蛋白 E, L209, native +11.18）

- 选型：训练集无 5O60 域（held-out），coverage=in（n_close 151）；核糖体 RNA 结合蛋白天然正电，是 RNA 类的典型代表。
- 两口径 H2（n30）：**big-global 4/5、小样本 4/5**（唯一败臂 p8 target+19 → +22, dev 2.7–3.3，深正过冲）。
- 数据：`output/generalization_ligand_v14_big/ligand/5O60_E/`、`output/generalization_ligand_v14_small/ligand/5O60_E/`、
  `output/charge_calibration_v14_5O60E_only.json`、`output/charge_calibration_v14_small_v2.json`（5O60_E per 条目 slope 1.433）。
- 意义：替换后 RNA 类别在 big-global 由"21KL_A/2E9R_X 0/5"变为"21KL_A 0/5 + 5O60_E 4/5"——
  天然正电 RNA 结合蛋白开箱可控，佐证 2E9R_X 的失败确系"长 × 负大电荷"而非 RNA 结合属性。

---

## 五、论文可引用表述（草拟）
> 模型在训练覆盖内的常规蛋白上电荷控制可靠（温和区 + 小样本标定后多档命中），
> 但对**长 × 大电荷变化（深负为主）组合的可设计性有限**：蛋白模式 1A65（L504, −26.9）
> 与配体模式 2E9R_X（L476, −10.1）两模式一致失败——前者深负欠冲（补数据仅 0/5→2/5 部分缓解），
> 后者负向过冲（big/small 0/5，per 大 slope 校准仍救不了负向臂）。两类失败方向相反，
> 但都发生在标定只能校正"可反推的增益偏差"、不能创造"模型未学会的生成"的弯曲饱和区。

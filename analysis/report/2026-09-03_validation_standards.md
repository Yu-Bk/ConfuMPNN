# v12.3/v14 验证重构 + 判据修正 主报告（2026-09-03）

> **本次任务**：① 覆盖核查统一标准 ② 口径框架修正 ③ 废除 72% 自我参照 ④ 校准表机理 + 分段拟合论证
> ⑤ 自适应反推规范 ⑥ 最新分组数据 ⑦ 历史错误更正+备份。详细过程 `session/2026-09-03_validation_refactor.md`。

---

## 一、本次任务统一标准（写入报告与对话，作为后续版本强制要求）

### 1.1 覆盖核查标准（工具 `code/tests/coverage_check.py`）
- 训练集 = 该版本实际训练 labels（v12.3 → `labels_v12_3_train.npz` 6580 域；v14 → `labels_v14_final.npz` 5371 域）
- 每验证蛋白统计：训练中"相近域"数 = 满足 |ΔL|≤max(0.15·L,40) 且 |Δq|≤4 的训练域数（q = net_charge@7.4）
- 判定：**≥100 = in**（训练覆盖内，标准验证集）；**30–99 = boundary**（覆盖边界，单独检）；**<30 = out**（分布外，泛化外推单独讨论）
- **不在训练覆盖内的蛋白不得放入标准验证集**；分布外蛋白单独跑、单独报告，不参与 H2/H1/Tm-Sol 达标判据。

### 1.2 口径框架（只保留两口径）
- **per-protein 表内校准 = 诊断阶段已对该蛋白做完整现场标定（乐观口径，剔除出达标讨论）**。诊断测了哪个蛋白，哪个蛋白就是"已标定"，其 per-protein 成绩不是"开箱即用"性能。
- 达标/报告只使用：
  - **big-global（纯训练域拟合，未标定）**：真"新蛋白开箱即用"水平
  - **小样本现场标定**（新蛋白现场采 ~50 条拟合自身 slope）：实操推荐口径
- 任何 global 校准表**不得混入 valid/测试蛋白的响应点**（否则对这批蛋白半泄漏）。

### 1.3 判据（废除自我参照）
- H2 科学判据 = **单臂 |实际平均电荷 − target| ≤ 2.0**（`DESIGN_CRITERIA.md`，物理目标，无歧义）
- **废除 "H2 ≥ 当前 72%"**：72% = v12.2 per-protein 自测成绩被误作后续版及格线，属循环参照。
  已更正 `PROJECT_LOCAL_V12_2.md` §7.5。
- 整体命中率按口径分列报告（big-global / 小样本），不做自我参照二分。

---

## 二、校准表机理备注（为什么分段拟合——长蛋白/深负蛋白的特殊性）

### 2.1 观测事实
- v12.3 失败蛋白全部"**欠负**"（target −13 生成 −7.7、−27 生成 −21.4），成功蛋白全近中性
- v12.3 响应曲线比 v12.2 更 S 弯曲（curvature 分析：bend 全升、intercept 更负、线性 RMSE 全升）
- v12.3 per slope 分布 **0.97–2.01**（差 2 倍），global 单线 slope 1.47 无法代表

### 2.2 长蛋白 / 深负蛋白的特殊性 → 对校准表的影响
1. **绝对电荷 target 对不同长度蛋白的"化学难度"不同**：净电荷是全局总量。
   同样 target=−25，对 150aa 蛋白需 net density −0.17（每个残基都偏负，化学上极端）；
   对 500aa 蛋白 density 仅 −0.05。但**达到深负都要插入大量 D/E**——对长蛋白，大量 D/E 会显著牺牲疏水核心/折叠（CE 损失惩罚非 native 序列）→ 生成侧有"物理阻力"。
2. **响应 S 弯曲 = CE/物化权衡的饱和**：温和区（target 接近 native±几）模型能跟随（近似线性）；
   深负/深正区，模型在"满足电荷"与"不破坏序列合理性"间权衡 → 实际电荷达不到 target → 曲线变平饱和 → S 形。
   **线性校准假设"生成 = slope·target + intercept"在弯曲段失效** → 反推的 target_eff 在极端区命中不了（这是 v12.3 深负欠冲在校准侧的机制）。
3. **intercept 负**：响应曲线在 target=0 附近整体下移（模型系统性倾向负/删正电的捷径响应），校准 intercept −5.9 反映此偏移。
4. **训练分布影响**（v12.3 特有）：补的 455 长蛋白**负电富集**（>400 域 mean_q −5.6、40% q<−10 vs 短域 27%/10%）→ ConditionEncoder 在负区的响应统计被重塑（见过更多负样本），但深负外推仍受生成侧阻力饱和 → 曲线形态改变，旧的线性/单斜率适配失效。

### 2.3 为什么分段拟合可行且必要
- 单斜率假设全曲线线性，在 S 弯曲下温和区与极端区**都不可能同时被一条线拟合准**。
- **分段线性**（如 target 按 native±5 / ±5~±15 / 更深 分段，各自拟合）→ 每段近似线性 → 反推准。
- **按蛋白分层**（长蛋白组/常规组各自拟合）→ 治"同长度内异质性"之外的组间差异（长蛋白 S 弯曲更早出现）。
- 前提：拟合分段要有覆盖该区段的诊断/标定点；对全新蛋白仍需小样本现场标定先取点。
- 深层含义：**校准只能校正"可反推的增益偏差"，不能创造"模型没学会的生成"**。若深负欠冲源于生成侧阻力饱和（模型权衡），分段校准可改善"已有能力范围内的命中"；若源于模型根本不会在深负区加够 D/E（训练没覆盖，见 §4 分布统计 q≤−20 仅 2.7%），则须训练侧补数据。两者要分清。

---

## 三、自适应反推（迭代反馈采样）规范
- **定义**：desired D → 按当前估计反推 target₁ → 采样 n → 实测 M₁ → 若 |M₁−D|>阈值，用实测修正增益 → target₂ → 再采样 → 收敛。
- **执行规范（防止覆盖训练收益 / 采样过久）**：
  1. **最多 3 轮**，每轮 n≤10（~30 条封顶），超 3 轮未收敛即停，接受残差或转小样本现场标定。
  2. **校准只改推理侧采样 target 反推，绝不改模型权重** → 不会覆盖训练学到的东西。
  3. **校准 vs 训练分工（回答"自动校准为何还要反复训练"）**：校准校正"平均电荷增益偏移"（推理侧）；
     训练决定"模型能否产生某电荷区段的**合理序列**（组成/折叠/物化）"。校准救不了"模型不会生成"——若反推 target 后模型仍生成不了（饱和），加轮次无意义。
     所以：**校准可自动，训练不可省**；自适应轮次限制就是为了避免误把"校准能修的"当"已解决"。
  4. 自适应反推与分段校准/小样本标定可叠加：小样本标定给初值，自适应收尾。

---

## 四、最新分组数据（重构验证集后）【待 agent 结果填充】
> in=标准验证集 / boundary=覆盖边界长蛋白单独检 / out=分布外泛化讨论。口径只报 big-global 与小样本。

### 4.1 v12.3 蛋白模式（对 6580 域训练集）
| 组 | 蛋白 | H2 小样本 | H2 big-global(纯训练域) | H1 TM≥0.7 | 备注 |
|---|---|---|---|---|---|
| in(5) | 1AZM/1AS2/2FEO/5CQH/1CGE | **20/25 = 80%** | 待 big 表补 | 25/25 | 标准验证集（1CGE 5/5、1AS2 4/5、1AZM 4/5、5CQH 4/5、2FEO 3/5）|
| boundary(2) | 1BJ4(L470)、13BB(L552) | 6/10 = 60% | 待补 | 10/10 | 长蛋白改进检验：**1BJ4 5/5（v12.2 small 也 5/5）**；13BB 1/5 欠负失败（native−12.8 臂 dev2.7、n2 dev2.9、n8 dev4.3 全欠负超阈值，仅 p8 命中）|
| out(2) | 1A65(native−26.9)、1CDG(L686) | 6/10 = 60% | 待补 | 10/10 | 分布外：**1A65 2/5（v12.2 small 0/5 改进）**、1CDG 4/5 |
| **合计** | 9 蛋白 | **32/45 = 71%** | 待补 | **45/45 = 100%** | 小样本口径略低于 v12.2(74%)，差异在单臂边界噪声内 |

- **H1 折叠：45/45 TM≥0.7（100%）**——9 单体（无二聚体），长蛋白 1A65/1BJ4/13BB/1CDG tm_median 0.90–0.97 全过。补长蛋白+40ep 未伤折叠。
- **Tm/Sol：S2 0/45 恶化**（vs 无条件基线；v12.2 0/50 同口径一致）。长蛋白 native 臂全部无恶化（1A65 ΔTm+6.8/Δsol+6.9、13BB −2.5/+1.7、1CDG −0.3/−0.1、1BJ4 −0.5/−0.5）。
- **小样本口径小结**：小样本现场标定能救 v12.3（32/45=71% vs v12.2 37/50=74%，差 1 臂边界噪声），深负长尾 1A65 较 v12.2 有改进（0/5→2/5）。
- ⚠️ **口径注**：H2 数据基于 **calib_small 批**（小样本标定）；**H1/Tm-Sol 基于旧 calib 批**（`generalization_v12_3_calib`，小样本批未重跑 ESMFold/Tm-Sol）——两批 recovery/GRAVY 一致（±0.01–0.03），折叠结论可平移，但严格应在 small 批复测。已通知 agent 补测（见 §7 待办）。
- ⚠️ **H2 计数修正**：agent 初报 33/45=73%（1A65 3/5）为误——其统计把混入 seqs.fa 的 native 序列计入 1A65 p2 臂（dev 2.13→1.999 虚翻命中）。权威口径（validation.json dev 或 seqs.fa 去 native）恒为 **32/45**。已核验 45/45 中仅此 1 臂受影响，其余 in/boundary 全组一致。
- 数据来源：`output/generalization_v12_3_calib_small/`（小样本 H2）、`output/generalization_v12_3_calib_stats.json`（H1，旧批）、`output/tm_sol_v12_3/tm_sol_summary.json`（旧批）。

### 4.2 v14 配体模式（对 5371 域训练集，RNA/DNA 占 7.7%）
| 组 | 蛋白 |
|---|---|
| in(10) | 6D2O/1AS2/2FEO/5CQH/1CGE/1BJ4/21KL_A/**5O60_E**/3MXB_A/9DWG_L |
| boundary(1) | 1A65（L504 超训练 max500 + 深负） |

> 🔄 **2026-09-03 晚方法学修正（2E9R_X 归档 → 5O60_E 替换）**：2E9R_X（FMDV RdRp, L476, native −10.1）不具 RNA"结合蛋白"代表性——核糖体 RNA 结合蛋白天然正电，2E9R_X 却是长负聚合酶，验证中系统性过冲（target −10 → 生成 −18.1，dev 8.1）。用户判定其属"**长×大电荷变化可设计性有限**"档（与蛋白模式 1A65 同类），**已从标准验证集移除归档**（数据保留：`output/generalization_ligand_v14{,_big,_small}/ligand/2E9R_X/` + `charge_calibration_v14_small.json` per 条目，新 manifest 统计自动排除）。**5O60_E（核糖体蛋白 E, L209, native +11.18, coverage=in n_close151, 训练集无 5O60 域=held-out）替换加入**作 RNA 结合蛋白典型代表。证据汇总见 `analysis/report/2026-09-03_long_neg_charge_limitation.md`。
> **计数注**：in 为 1:1 置换仍 10 项；任务文本的"in-9 参考值"= 剔除 2E9R_X、尚未计入 5O60_E 的中间口径（big 22/45=49% / small 36/45=80% / per 40/45=89%），下表中同时给出。in manifest = `validation_manifest_v14_in.json`（10 项含 5O60_E）。

**① 配体诊断 slope（未校准）**：valid 10 蛋白 slope 均值 **1.49 ± 0.40**（响应增益，需校准）；
trainish 8 域同网格（供对照）。global 线性校准（216 点）：slope 1.500 / intercept −1.764。（诊断网格基于原 in-10，未含 5O60_E）

**② H2（per-arm |dev|≤2，in 10 × 5 臂）——按口径分列（5O60_E 两口径均 4/5）**：
| 口径 | H2（in 10 含 5O60_E） | in-9 中间口径（剔 2E9R_X） | 说明 |
|---|---|---|---|
| **big-global**（纯训练域 60 域 300 点，开箱） | **26/50 = 52%** | 22/45 = 49% | 成功蛋白 6D2O/1AS2/3MXB_A 5/5；**RNA 代表 5O60_E 4/5**（天然正电开箱即好，仅 p8 target+19 过冲 dev 3.3）；21KL_A 0/5（正电过冲）；1CGE/5CQH/1BJ4/9DWG_L 各 1/5 |
| **小样本现场标定**（5 target×n10 拟合，n30 测） | **40/50 = 80%** | 36/45 = 80% | 5O60_E 4/5（仅 p8 dev 2.7）；剔除 2E9R_X 后无全败项；9DWG_L 1/5 仍弱（LOOCV-unreliable） |
| per-protein 表内（诊断网格，剔除出达标） | 40/45 = 89%（9 网格蛋白；5O60_E 无诊断 per 条目，不入此口径） | 40/45 = 89% | 仅参考 |

**③ boundary 1A65（big-global 开箱，单列）**：**0/5**（native −27→−24.5，dev 2.0–3.3，深负长蛋白欠冲 2–3 电荷）。native_q −26.85 落在训练 q 分布 2.8% 分位 + L504 超训练 max500 → 边界预期。

**④ 组成（native 臂带电总数倍率 gen/native，0.7–1.3 判据）——❌ A1-global 未根治删减（2E9R_X 已归档剔除）**：
| 蛋白 | 6D2O | 1AS2 | 2FEO | 5CQH | 1CGE | 1BJ4 | 21KL_A | 3MXB_A | 9DWG_L |
|---|---|---|---|---|---|---|---|---|---|
| ratio | 0.56 | 0.46 | 0.46 | 0.43 | 0.60 | 0.46 | 0.61 | 0.69 | 0.47 |
in(9，原数据) 0.43–0.69，**系统性删减未解除**（与 v13 0.55–0.69 同级甚至略低）；RNA/DNA 蛋白亦删（21KL_A 0.61 / 3MXB_A 0.69；2E9R_X 0.44 随归档剔除）。
→ A1 全局化（floor0.8/λ0.3/normalize）**未奏效**：期望计数监督 vs 离散采样 gap + 删减先验仍主导；Tm/Sol 需警惕（与 v13 同因）。
> ⏳ 5O60_E 组成/H1/Tm-Sol 未在本替换任务内重跑（其 per 口径采样与 ESMFold 链属后续验证推进），现仅有 big/small 两口径 native 臂生成序列（`generalization_ligand_v14_{big,small}/ligand/5O60_E/`）。

**⑤ H1/H3/Tm-Sol**：⏳ ESMFold 回折运行中（partial ~20/50 臂），H3/Tm-Sol 待链推进，完成后补填（注：新 in manifest 含 5O60_E，H1/Tm 链需对 5O60_E 补跑，其两口径采样产物已在 `generalization_ligand_v14_{big,small}/ligand/5O60_E/`）。数据源：`output/generalization_ligand_v14{,_big,_small}/`、`output/v14_ligand_diag_response.json`、`output/charge_calibration_v14_{big,small,small_v2}.json`、`output/charge_calibration_v14_5O60E_only.json`、`output/v14_ligand_comp.json`、`output/v14_ligand_gen_stats{,_big,_small}.json`。

> ⚠️ H2 计数口径：一律读 validation.json 的 dev 字段（seqs.fa 已含 1 条 native，不直接对全 seqs.fa 求均值）。H1/Tm 用 per-protein 批（OUT）序列。

### 4.3 v12.2 同标准对照
- **v12.2 覆盖（相对 v12.2 训练 6710，实测 coverage_check）**：in = 1C6O/1AG0/1AS2/1AZM/1CGE/2FEO/5CQH（7）；out = 1A65(n_close 4)/1AXW(7)/**1BJ4(n_close 20)**。
  注：v12.3 补长蛋白后 **1BJ4 n_close 20(out)→96(boundary)**——补长蛋白把 1BJ4 拉进覆盖边界。
- **v12.2 两口径基准（旧 10 蛋白 manifest，含 3 二聚体）**：小样本 37/50=74%；big-global(noleak) 22/50=44%。
  v12.2 小样本下 1A65 仍 0/5（标定救不了深负长尾）；1BJ4/2FEO 5/5（被小样本救活）。

**共同 7 蛋白逐蛋白两口径对比（小样本）**：
| 蛋白 | v12.2 small | v12.3 small | 变化 |
|---|---|---|---|
| 1AZM | 3/5 | 4/5 | +1 |
| 1AS2 | 5/5 | 4/5 | −1 |
| 2FEO | 5/5 | 3/5 | −2 |
| 5CQH | 5/5 | 4/5 | −1 |
| 1CGE | 5/5 | 5/5 | = |
| 1A65 | 0/5 | 2/5 | **+2（深负长尾改进）** |
| 1BJ4 | 5/5 | 5/5 | = |
| 小计 | 28/35=80% | 27/35=77% | −1（1 臂边界噪声：1A65 p2 dev2.13 差一点） |

**big-global（纯训练域）**：v12.2 同 7 蛋白 18/35=51%（1A65 0、1BJ4 1、2FEO 3…）；v12.3 big 表构建中 → 补填。
- 新增（仅 v12.3 有）：13BB small 1/5、1CDG small 4/5（均训练外/边界）。
- **结论（覆盖内小样本同口径）**：v12.3 共同 7 蛋白 77% vs v12.2 80%，**略降 1 臂**（1A65 p2 边界臂 dev2.13 未命中；若取 8 分位内差异属噪声）；v12.3 的增量收益在**深负长尾 1A65（0/5→2/5）**，代价是 in 组 2FEO/1AS2/5CQH 各降 1-2 臂。

---

## 五、历史错误更正 + 备份
> 原则：不再让读者两文件跳转对比；错误点/响应数据/结论直接在本报告更正，旧版本数据备份在本报告末尾。

### 5.1 需更正的旧结论（立即）
- "v12.3 校准后 H2 49%/40% 未达标" → **误判**（混入分布外 + 用表内口径）。更正见 §4.1（覆盖内结果）。
- "v12.2 H2 72% 为达标成绩" → 该 72% 是 per-protein 表内（诊断=已标定）+ 含 3 二聚体旧验证集口径，**非开箱即用**。v12.2 真未见 = big-global 44%。
- 旧校准表（`charge_calibration_v12_3.json` 的 global）混入 valid 点 → **作废 global 字段**，改用纯训练域 big 表（§四数据）。

### 5.2 备份区（旧错误数据，仅供追溯）
**v12.3 蛋白模式旧数据（作废，勿用于判据）**：
- 旧 H2（混入分布外 + 表内口径）：per-protein 22/45=49%；旧含泄漏 global（charge_calibration_v12_3.json global 混 9 valid 响应点）40% → 均作废。
- 作废校准表 global 参数：`charge_calibration_v12_3.json` global slope 1.471 / intercept −5.935（含 valid 泄漏）；v12.2 表 global 1.579 / −3.139（同含 valid 泄漏）。真未见 global 用纯训练域 big 表（v12.3 `charge_calibration_v12_3_big.json`、v12.2 noleak）。
- v12.2 旧验证集 10 蛋白（含二聚体 1C6O/1AXW/1AG0）的 per-protein 72% = 乐观口径（诊断=已标定），仅参照非判据。
- v12.3 早期响应诊断 global slope 对比（未校准响应增益改善证据，仍有效）：valid slope 1.496 vs v12.2 1.562；curvature 分析见 `2026-09-02_v12_3_curvature_analysis.md`。

---

## 六、历史报告影响盘点
> 按用户要求：**所有任务完成后再做**，统一标注哪些历史结论受旧验证集/旧校准表影响。

（待 §四数据齐全后执行）

---

## 七、待办（v12.3 agent 2026-09-03）
1. **calib_small 批补测 H1 ESMFold + Tm/Sol**（§4.1 口径注）：
   - Tm（TemBERTure 45 arm）+ Sol（Protein-Sol 45 arm）已启动（log `v12_3_tm_small.log`/`v12_3_sol_small.log`）；uncond/native 基线复用 `tm_sol_v12_3/`（同模型同 manifest，跨批一致）。
   - ESMFold H1（45 arm）等 GPU 空档补跑（`esmfold_score.py --input-dir generalization_v12_3_calib_small`），产物 plddt.csv → tm_score → 更新 §4.1 H1 列为 calib_small 批权威值。
   - 若两批结果一致（预期，recovery/GRAVY 已证可平移）则保留旧批结论并标注已验证。
2. **big-global 纯训练域表**：`charge_calibration_v12_3_big.json`（100 训练域×5×n10）GPU6 构建中 → 用其 resample 9 蛋白 global 口径 H2（**判命中用 validation.json dev 或 seqs 去 native，勿混入 native**）→ 补 §4.1/4.3 big-global 列。
3. H2 统计统一规范：**seqs.fa 末尾含 native 参考行，统计前必须跳过（name.startswith('seed_') 或读 validation.json dev）**——1A65 p2 曾因混入 native 虚翻命中 1 臂（33/45→权威 32/45）。

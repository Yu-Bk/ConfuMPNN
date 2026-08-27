# P1 对照实验计划（v3 方案 §8 P1 执行细化）

> **定位**：v3 论文导向方案（`index/PROJECT_LOCAL.md`）P1 阶段的**对照实验执行模板**。
> 本文件既是本轮执行依据，也是**后续 P2/P3/P4 阶段对照实验的可复用参考模板**。
> 状态：计划（等待 v10 定位决策 + v10 训练完成后执行）。
> 日期：2026-08-27。

---

## 0. 前置前提（执行对照实验前必须满足）

> **关键原则**：对照实验的结论只在"模型与代码基线稳定"的前提下才可信。
> 因此执行顺序为：
> ```
> P0 代码改造（已完成） → v10 训练（基于 P0 代码） → v10 泛化验证
>     → 对照实验 C1–C4/C6 → PROPKA 复核(H4) → AF2 子集 → 统计汇总
> ```

**基线稳定性要求（全部满足才跑对照）**：
1. ✅ v10 训练脚本可跑（`train_finetune.py` 新增 A/B/C 三开关）——冒烟 50 域通过
2. ⏳ v10 正式训练完成（30 epoch，MoMPNN 与 LigandMPNN 双 backbone）
3. ⏳ v10 泛化验证通过（判定标准沿用 v3 §6：H1a TM 中位≥0.70 / H2 dev≤2.0）
4. ✅ 数据/打分管线就绪（ESMFold + US-align + Protein-Sol + TemBERTure + PROPKA + freesasa）

**v10 训练与 P0 代码的关系**：P0 是代码基础设施（target 自动补全 / RMSD 联报 / PROPKA 脚本 / fractional SASA / pH 自适应过滤器），**v10 训练直接跑在 P0 之后的代码上**。P0 的效果验证**不做训练前单独回归**——v10 能训练出并验证通过，本身就是 P0 代码正确性的证明；P0 各项改动的效果在 v10 的验证数据里统一体现。

---

## 1. 统一协议（所有对照实验共用）

| 项 | 设置 |
|----|------|
| 测试集 | 泛化 10 蛋白（`data/validation_pdbs/`，已有 manifest + 防泄漏）|
| 电荷臂 | native / −2 / +2 / −8 / +8（5 臂）|
| pH | 7.4（P1；多 pH 网格属 P4）|
| n | ≥30/臂（PROPKA/AF2 用 5–10 子集）|
| seed | 固定 seed，**对称配对采样**（同 randn → 同解码顺序，唯一差异=实验变量）|
| 打分 | ESMFold 回折 → US-align TM+RMSD；Protein-Sol %sol；TemBERTure Tm；GRAVY；pairwise identity |
| 判据 | H1a TM 中位≥0.70 且失败率≤10%；H1b RMSD 中位（按域报告）；H2 dev≤2.0；H4（PROPKA）dev≤2.0 |

**对称配对采样是本计划的方法论核心**：让两组对比只差"实验变量"，排除采样噪声（沿用 v9 已验证的 `--n 30 --seed 固定` 协议）。

---

## 2. 实验总览

| # | 对比项 | 检验的 claim | 数据来源 | 图表 |
|---|--------|--------------|---------|------|
| C3 | MoMPNN-only vs **MoMPNN+v10** | 显式条件 vs 隐式偏好（**核心**）| 需补跑 | Fig.4 上 |
| C1 | backbone-only vs backbone+v10 | 条件化绝对增益 | C3+补跑 | Fig.3 |
| C4 | LigandMPNN vs LigandMPNN+v10 | 配体模式条件化增益 | C1+现有 | Fig.3/4 |
| C2 | 引导采样 vs 条件微调 | 推理时引导 vs 训练时条件 | 需补跑 | 权衡散点 |
| C6 | 条件化 vs 无条件的 sol/tm/pLDDT | 条件化的"代价"（P7）| C1/C3 复用 | Fig.4 下 |
| H4 | Q_design vs Q_phys（PROPKA）| 物理真实性（P5）| folds/ 复用 | Fig.6 |
| AF2 | ESMFold vs AF2 | 独立第二验证器（P6）| 需补跑 | 对比表 |

---

## 3. 实验详情（模板卡片）

### 3.1 C3（最高优先）：MoMPNN-only vs MoMPNN+v10 —— 核心 claim
- **对照谁**：DPO 微调 backbone（不接受显式指令）vs 显式电荷条件控制（+v10 编码器）。
  - 这是"隐式偏好对齐（ProtAlign/MoMPNN 路线）vs 显式连续条件控制（本项目路线）"的直接对决。
- **采样**：10 蛋白 × 5 臂 × n30 × pH7.4。对称配对（同 seed）。MoMPNN-only 用 `--cond_encoder None` 直采；+v10 用 v10-MoMPNN 编码器。
- **数据量**：10×5×30×2 = **3000 条**（补跑）。
- **统计**：每臂 H2 命中率 + dev 分布；配对 Wilcoxon。
- **图表**：Fig.4 上半——两路线电荷可达性（达标率 + 电荷箱线图）。

### 3.2 C1：backbone-only vs backbone+v10（通用增益）
- **对照谁**：无条件 backbone vs 条件化（MoMPNN 与 LigandMPNN 各一套）。
- **采样**：C3 覆盖 MoMPNN 侧；LigandMPNN-only 补跑 10×5×30 = **1500 条**。
- **图表**：Fig.3（泛化达标率 + TM/RMSD）。

### 3.3 C4：LigandMPNN vs LigandMPNN+v10（配体模式）
- **对照谁**：C1 在配体模式特化；验证 v10 配体上下文条件化增益（复用 C8 配体消融）。
- **数据**：C1 的 LigandMPNN-only + 现有 v10 配体数据。
- **图表**：并入 Fig.3/4。

### 3.4 C2：引导采样路线 vs 条件微调路线
- **对照谁**：同一"控制电荷"目标的两种工程实现——推理时 logit-bias 引导（`run_guided.py` 无编码器 + `make_dynamic_callback`，含 P0-5 pH 自适应过滤器）vs 训练时条件注入（v10）。
- **采样**：10 蛋白 × 5 臂 × n30 = **1500 条**（补跑）。
- **图表**：X=电荷命中率、Y=TM 中位，两路线的权衡散点（Pareto）。

### 3.5 C6：条件化的"代价"（P7）
- **对照谁**：同蛋白同臂，backbone-only vs +v10 的 %sol / Tm / pLDDT。
- **数据**：复用 C1/C3 两组序列，补打 Protein-Sol + TemBERTure（**无需重新采样**）。
- **统计**：配对 Wilcoxon / 配对 t；Δ 分布。
- **图表**：Fig.4 下半——%sol/Tm/pLDDT 的 Δ 箱线图。

### 3.6 PROPKA 复核（H4，P5）
- **对照谁**：设计电荷 Q_design（游离 pKa）vs 物理修正电荷 Q_phys（PROPKA3 微环境 pKa）。
- **采样**：现有 folds/（ESMFold 回折结构已存在），每臂抽 n5–10 → 10×5×5 ≈ **250 条**。
- **工具**：`code/tests/propka_charge_check.py`（P0-3 已就绪）。
- **图表**：Fig.6——Q_design vs Q_phys 散点（对角线）+ H4 命中率表。

### 3.7 AF2 交叉回折（P6）
- **对照谁**：ESMFold 结论在独立第二验证器 AF2 下方向是否一致。
- **采样**：10 蛋白 × 3 代表臂（native/−8/+8）× n5–10 ≈ **150–300 条**，apo 回折（不含配体）。
- **图表**：AF2 pLDDT + TM/RMSD vs ESMFold 对比表。

---

## 4. 数据量充足性评估

| 图表 | P1 能否形成 | 说明 |
|------|-----------|------|
| Fig.3（泛化达标率+TM/RMSD）| ✅ | 补跑无条件对照 + 全量打分后即可 |
| Fig.4（C3 核心 + C6 代价）| ✅ | 补跑 MoMPNN 侧 + %sol/Tm |
| Fig.6（Q_design vs Q_phys）| ✅ | PROPKA 批跑 folds/ |
| Fig.2（controllability 矩阵）| ❌ → P4 | 需多 pH 网格 |

- **趋势图/箱线图**：n30 × 10 蛋白 = 80 数据点/指标，足够。
- **统计显著性**：10 蛋白对配对检验偏弱 → P1 出"趋势 + 初步统计"，**P4 扩 50 骨架**出正式主图。
- **新增量**：6000 条采样（GPU ~2–3h）+ ~8400 条 %sol/Tm 打分（CPU 数小时）+ 250 PROPKA + 150–300 AF2。

---

## 5. 执行顺序与算力（预算 5–6 天，AF2 并行）

| 步 | 内容 | 用时 |
|----|------|------|
| 0 | v10 训练（MoMPNN + LigandMPNN）+ 泛化验证 | 前置（1–2 周）|
| 1 | `validate_generalization.py` 加 `--uncond` 开关 + v10 backbone 分支 | 0.5 天 |
| 2 | 补跑 C3 + C1-LigandMPNN-only + C2 = 6000 条 | 1 天 |
| 3 | ESMFold 回折 + tm/plddt + %sol/Tm 全量打分 | 1.5–2 天 |
| 4 | PROPKA 复核 → H4 表 | 0.5 天 |
| 5 | 统计脚本（bootstrap CI / 配对检验）+ 汇总 JSON | 0.5 天 |
| 6 | AF2 子集 | +1 天（与 3–5 并行）|

---

## 6. 待确认事项

1. **v10 定位**：论文主方法（替换 v7/v9）vs 改进型消融（v3 §11 遗留）——决定对照基础。
2. **v10 双 backbone 范围**：本计划默认 v10 同时覆盖 MoMPNN + LigandMPNN（A/B/C 方法升级 backbone 无关，两场景都需验证）；若算力有限可先做单 backbone。
3. **C2 保留与否**：与 C1 有部分重叠（都是无条件 vs 条件），多一个"logit-bias 引导"工程变体；可省 1 天。
4. **公开策略**：数据子集 / 新 Release / Docker（v3 §11）。

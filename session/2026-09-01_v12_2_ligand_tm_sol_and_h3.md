# 会话记录 — 配体 Tm/Sol 物化验证 + 删减根治设计 + H3 准备（2026-09-01）

> **状态**：后台任务运行中（Tm/protein-sol），A1+A2+开关设计定稿并已写计划，H3 脚本就绪待跑。
> **关联**：根因 `analysis/report/2026-09-01_v12_2_ligand_comp_analysis.md`；设计 `PROJECT_LOCAL_V12_2.md §7`；
> 配体验证链 `analysis/report/2026-09-01_v12_2_ligand_validation.md`。

## 一、目标与顺序（用户明确要求）

1. **先跑配体 Tm/Sol**，看结果（删减的物理后果）
2. **更新 A1+A2 + keep/free 开关计划**——重点设计两点：① pocket 一定是**温和更改**；② pocket 定义
   与 core 定义**不冲突无矛盾 bug**
3. **最后跑 H3**（mompnn + ligand 双线），并判断是否需要历史版本对照

## 二、配体 Tm/Sol 流程（进行中）

### 2.1 流程与产物（`output/tm_sol_ligand_v12_2/`）

| 步骤 | 内容 | 状态 |
|------|------|------|
| uncond 采样 | 10 蛋白 × 30 条（配体上下文、net_charge=训练均值 1.4243，GPU6）| ✅ 10/10 |
| Tm | TemBERTure 3-replica CPU（泛化 arm+native_ref 60 + uncond 10 = 80 fasta）| ⏳ 泛化 ~10/60，uncond 启动 |
| protein-sol | perl 版（泛化 50 + native_ref 10）| ⏳ ~40/50 |
| 汇总 | `v12_2_ligand_tm_sol_summarize.py` | 待 Tm/sol 完成 |

**要点**：
- **native_ref fasta 来源**：泛化 `arm_native/seqs.fa` 末尾 native 行提取（与采样同序列空间）
- **protein-sol 必须串行**：`protein_sol_predict.py` 共享工作目录（固定 `input.fasta` 名），并行会互相覆盖
- **Tm 用 confumpnn-temberture 环境 + HF_HUB_OFFLINE=1**

### 2.2 脚本拆分（用户要求：MoMPNN 与 LigandMPNN 分开保存）

| MoMPNN 版 | LigandMPNN 版 |
|-----------|---------------|
| `sample_unconditioned.py`（蛋白模式，已回退 --mode 改动）| `sample_unconditioned_ligand.py`（ligand featurize 硬编码）|
| `v12_2_tm_sol_summarize.py`（原样）| `v12_2_ligand_tm_sol_summarize.py`（配体路径 + **排除 native 参考行**，避免污染均值）|

## 三、A1+A2 + keep/free 开关设计（定稿，`PROJECT_LOCAL_V12_2.md §7`）

### 3.1 核心：三块互斥残基分区（解决 pocket vs core 矛盾 bug）

**v12 现状的矛盾**：`surface_charge_target_loss` 用 `core_mask=(~surf)` 锁死核心（q_core 用 native
one-hot 算）。深部口袋 frac_sasa<0.25 被划入"核心"。若把口袋直接并入表面 mask，同一残基**既在 q_core
（native 值）又在 q_surf（生成值）→ 双算 → 模型改口袋时总电荷 drift 且监督看不见**。

**解法**：每残基只属一块，无重叠：

| 分区 | 定义 | 行为 |
|------|------|------|
| core（锁死）| frac_sasa<0.25 且距配体≥8Å | q_core = native one-hot 锁死（现状）|
| pocket（温和改）| 距配体<8Å（无论 frac_sasa）| 净电荷锚 + A1 双向计数 |
| surface（温和改）| frac_sasa≥0.25 且非口袋 | 现状表面监督 |

三块互斥 → q_core 不再含口袋残基 → pocket 生成电荷全部进入 q_surf 监督 → 总电荷恒 = target。

### 3.2 pocket "温和更改"量化定义（保量不保位，≠fix）

1. **净电荷**：pocket∪surface 锚到 `target − q_core`
2. **总数（A1 双向）**：`relu(N_p×0.7 − gen) + relu(gen − N_p×1.3)`，D/E 与 K/R 双计数
   - floor 0.7 堵配体删减（0.53-0.65 触发）
   - **ceil 1.3 防成对加**（v12 只设下限无上限 → 过度添加 1.5-2× 的教训）
3. **具体位置**：完全自由

### 3.3 keep/free 开关

- `--pocket_mode keep`（默认）：pocket 带电总数+净电荷受保护 → 保/加强配体结合
- `--pocket_mode free`（可选）：推理侧不传保护（模型默认倾向 = 原版配体疏水先验），零训练成本
- 二期：训练注入 pocket 保护占位符 flag（类 S3），模型学会 keep/free 双语义

## 四、H3 脚本（就绪，待执行）

`code/tests/h3_charge_legality.py`：structure_aware_filter 4 规则（charge_cluster / salt_bridge /
core_charge / same_sign_cluster）**全量事后统计**。要点：
- **compute_bias 不适用**：它只统计"未解码可抑制位置"（seq_int==20），完整序列下全 0 → 须独立实现
- 坐标 = ref 骨架 Cα；带电集合 = `pH_adaptive_charged_aa(7.4)` = K/R/D/E
- 基线① native_ref ② 无条件基线（训练均值占位）
- 判据：条件臂违规率 ≤ max(native, uncond) + 0.05

**对照判断（不需要历史版本）**：
- native_ref + 无条件基线是同骨架最恰当对照（隔离"电荷条件化额外代价"）
- mompnn/ligand 双线已覆盖"有无删减"维度
- 历史版本重采样成本高、信息增量低；若配体线 H3 意外失败再补 v12.1 定位

## 五、迁移收尾决策框架（等 Tm/Sol + H3）

| 情形 | 结论 |
|------|------|
| Tm/Sol 无恶化 **且** H3 双线过 | 路径 A：快速定稿——删减如实报告为已知局限 + fix 缓解；论文避免"配体结合能力保持"类声明 |
| Tm/Sol 恶化 或 H3 配体超标 | 路径 B：执行 §7 A1+A2 重训（~16h）→ 组成健康 → 全链复验 |

**"完成迁移"判据**：不是单个性质 OK，而是验证链全绿（H1/H2/H3/H4 + Tm/Sol + 组成）。当前缺
Tm/Sol（测中）+ H3（未跑）+ 组成（删减未解决）。

## 六、后台任务记录

| 任务 ID | 内容 | 状态 |
|---------|------|------|
| bb9g7rgna | 配体 uncond 采样（GPU6）| ✅ 完成 10/10 |
| bau99ay3p | Tm 泛化 arm + native_ref（60 fasta）| ⏳ 运行中 |
| bfmkxwg7q | Tm uncond（10 fasta）| ⏳ 运行中 |
| bomlksnwm | protein-sol 泛化 + native_ref（60 fasta）| ⏳ 运行中 |

**复现命令**（项目根）：
```bash
# uncond 采样
PYTHONPATH=code python code/tests/ligand_v9/sample_unconditioned_ligand.py \
  --manifest data/validation_pdbs/validation_manifest.json \
  --out_dir output/tm_sol_ligand_v12_2/uncond \
  --cond_encoder output/finetune_ligand_v12_2/finetune_epoch030.pt \
  --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
  --n 30 --device cuda:6 --pH 7.4
# Tm（confumpnn-temberture）
python code/tests/temberture_score.py --input-dir output/tm_sol_ligand_v12_2/seqs
python code/tests/temberture_score.py --input-dir output/tm_sol_ligand_v12_2/uncond
# protein-sol（串行）
for fa in $(find output/generalization_ligand_v12_2/ligand -name seqs.fa); do
  python3 protein_sol_mcp/scripts/protein_sol_predict.py "$fa"
done
# 汇总
PYTHONPATH=code python code/tests/ligand_v9/v12_2_ligand_tm_sol_summarize.py
# H3 双线
PYTHONPATH=code python code/tests/h3_charge_legality.py --gen-root output/generalization_v12_2_calib_small/protein \
  --ref-root output/generalization_v12_2_calib_small/ref \
  --native-root output/tm_sol_v12_2/ref_native --uncond-root output/tm_sol_v12_2/uncond \
  --pH 7.4 --out output/h3_protein.json
PYTHONPATH=code python code/tests/h3_charge_legality.py --gen-root output/generalization_ligand_v12_2/ligand \
  --ref-root output/generalization_ligand_v12_2/ref \
  --native-root output/tm_sol_ligand_v12_2/ref_native --uncond-root output/tm_sol_ligand_v12_2/uncond \
  --pH 7.4 --out output/h3_ligand.json
```

# 会话记录 — v13 配体删减根治重训（A1+A2 执行）（2026-09-01）

> **状态**：v13 配体重训运行中（GPU6，30ep，任务 bteew2epu）。
> **关联**：物化验证报告 `analysis/report/2026-09-01_v12_2_ligand_tm_sol_h3.md`；
> 设计 `PROJECT_LOCAL_V12_2.md §7`；根因 `2026-09-01_v12_2_ligand_comp_analysis.md`。

## 一、决策链：Tm/Sol + H3 → 路径 B

用户执行序列（前序）→ Tm/Sol → A1+A2 计划 → H3 双线 → 迁移收尾决策。

### 1.1 配体 Tm/Sol 物化验证（未通过）
- **S2：9/50 臂明显恶化（Tm Δu < −5℃），8/9 为负电臂（n2/n8），正电臂 0**
  - 5CQH n8 −9.02 / n2 −7.35 / native −5.66；2FEO n8 −7.64 / n2 −5.85；1AG0 n8 −6.68；
    1AS2 n8 −5.64 / n2 −5.11；1C6O n8 −5.53
  - **对照 mompnn 蛋白线 v12.2：S2 0/50** → 恶化是配体微调引入、定向在删减发生处
- Sol：无恶化（负电臂 sol 大涨 = 删带电残基的组成信号，非物理变好）
- 机理解释：删带电残基 → 破坏盐桥/表面极性 → 热稳下降；净电荷要求越极端删越多

### 1.2 H3 电荷合法性（未全绿）
| 线 | 通过 | 失败臂 |
|----|------|--------|
| mompnn | 48/50 | 1C6O/n8（0.297 vs 0.242）、1A65/n8（0.217 vs 0.207）|
| ligand | 46/50 | 1C6O/n8、1A65/{native,n2,n8}（基线过紧 uncond 0.076）|

共同失败 n8（删电荷重排 → R4 same_sign_cluster 大幅升高）。H3 与 Tm 同源（删减）。

### 1.3 决策（用户确认）
Tm/Sol 恶化 + H3 配体超标 → **路径 B：A1+A2 重训治删减**。报告 `2026-09-01_v12_2_ligand_tm_sol_h3.md`。

## 二、A1+A2 实现（commit 1b93e87）

### 2.1 `code/src/v12_losses.py`
- 新增 `pocket_count_loss(logits, pocket_mask, native_pocket_counts, floor=0.7, ceil=1.3)`：
  D/E 与 K/R 双向计数 `relu(N·floor−gen) + relu(gen−N·ceil)`（floor 堵删减 / ceil 防成对加）
- `surface_charge_target_loss` 加 `extra_mask` 参数：监督 mask = surface ∪ extra_mask（A2）

### 2.2 `code/train_finetune.py`
- 超参：`--pocket_mode keep|free`（默认 keep）、`--pocket_cutoff 8.0`、
  `--pocket_floor 0.7`、`--pocket_ceil 1.3`、`--lambda_pocket 0.2`
- 每域三块互斥分区（v12_supervision 块内，`--pocket_mode keep` 时）：
  - pocket = Cα-配体最近距离 < 8Å（`dom["Y"]` 配体原子，define_pocket.py 口径）
  - core = frac<0.25 且非 pocket；surface = frac≥0.25 且非 pocket
  - `dom["pocket_mask"] / core_mask / charge_surf_mask`（= surface ∪ pocket）
- 损失：
  - v12_ct：core_mask 改用互斥 core，`surface_charge_target_loss(extra_mask=charge_surf_mask)`
    → pocket 生成电荷进入监督，核心不再含 pocket → 无双算/无 drift
  - 新增 pocket 损失：`total += λ_pocket · pocket_count_loss`
- checkpoint 追溯字段 + NaN 监控加 pocket/ct 分量

### 2.3 修复的 bug
1. **Cα 索引**：`dom["X"][0, 1]` 取到单残基 4 原子 `[4,3]`（应为 `X[0, :, 1]` 全部残基 Cα `[L,3]`）
   → surf(282)+pocket(4) 广播错被 except 误报为 "SASA 计算失败"
2. **H3 脚本**：ref 骨架 PDB（REMARK "reference skeleton from parse_PDB"）**resnum 全归一化为 4**
   → LigandMPNN parse_PDB 的 CA_dict（依赖 resnum 唯一性）塌缩为 1 残基 → 越界
   → 改为按行序直接提取 CA 坐标（坐标几何正确，文件顺序=残基顺序）
3. **汇总脚本**：native_ref Tm csv 行名 `<PDB>_native L=<L>` 非 `seed_` 开头被排除
   → `read_tm_csv(seed_only=False)` 读 native_ref

## 三、启动前检查 + dry-run + 训练启动

| 检查项 | 结果 |
|--------|------|
| 计划 | v12.2 配体命令 + A1+A2 超参（floor0.7/ceil1.3/λ0.2）|
| 数据 | symlink 0 dangling、4957 域 labels、PDB 含 HETATM（101M 192 个）|
| 环境 | GPU6 空闲（3% 利用率）、权重在、绝对路径 python |
| dry-run | 50 域 1ep：0 NaN、0 分区失败、50/50 三块互斥分区正常 |
| 启动确认 | 2 分钟后进程存活、GPU6 11.9GB/100%、预解析推进 |

**训练命令**（`log/v13_ligand_train.log`，后台任务 bteew2epu）：
```bash
PYTHONPATH=code nohup ~/miniconda3/envs/confumpnn/bin/python code/train_finetune.py \
  --ligand --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
  --labels data/ligand_train/labels.npz --dompdb data/ligand_train/all_pdb \
  --out_dir output/finetune_ligand_v13 --device cuda:6 --epochs 30 \
  --decouple_absolute --decouple_abs_lo=-35 --decouple_abs_hi=20 \
  --v12_supervision --frac_floor 0.5 --gravy_margin 0.4 --lambda_v12 0.2 \
  --lambda_target 0.2 --sasa_threshold 0.25 \
  --ph_aware_filter --structure_boost 1.5 \
  --charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15 \
  --lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 \
  --pocket_mode keep --pocket_cutoff 8.0 --pocket_floor 0.7 --pocket_ceil 1.3 --lambda_pocket 0.2 \
  --log_file log/v13_ligand_train.log
```

## 四、复验链（训练完成后）

1. **组成分析**：泛化 10 蛋白带电残基总数 vs native（判据 0.7-1.3×，§7.5）——**核心验证删减被治**
2. **配体诊断 slope**（17 蛋白，校准后 valid 区内 ∈ [0.9,1.15]）
3. **泛化采样 n 50**（10 蛋白 × 5 臂，⚠️ 用户 2026-09-01 决策：n30→50 扩大样本量）
4. **泛化 H2**（per-protein 校准）+ **H1 折叠**（ESMFold TM ≥ 0.7）+ **H4 PROPKA**
5. **H3 双线复测**（**全臂 × n50 统计，不只 n8**——单臂有偶然性；判据不变 ≤ 基线+5pp）
6. **Tm/Sol 复测**（负电臂 Tm 应恢复，S2 判据，同批 n50 序列）

判定：组成健康 + slope 达标 + H2 ≥ 当前 72% + H3/Tm/Sol 全绿 → 迁移 v9 完成。

## 五、决策记录（2026-09-01 用户讨论，已固化 PROJECT_LOCAL_V12_2 §7.7）

**Q：条件化采样当前无结构过滤器 bias，是否严重？加 bias 补丁？**

- **现状**（实证）：防聚集规则（charge_cluster/salt_bridge/core_charge/same_sign_cluster）只以
  两种形态存在——训练侧 C 组件 `ph_aware_structure_penalty`（损失、软惩罚，v12.2/v13 都开）+
  Phase 1 引导模式的 logit bias（`make_dynamic_callback`）。**条件化采样 `conditioned_sample`
  不传 bias_callback → bias=0**（2026-08-29 bias 排查实证）。
- **严重性**：不是"完全无防护"（训练侧软惩罚兜底，H3 92-96% 通过），但极端负电臂（n8）因删减
  捷径逃逸 → 聚集违规。根因=删减，v13 A1 从训练侧堵。
- **决策（用户确认）**：
  ① **扩样本量**：H3 判定不只 n8（n30 有偶然性）→ v13 泛化采样 n50、全臂统计；
  ② **bias 补丁=零成本可选保险，⚠️ 非必选、默认不加、不进复验链**（用户强调"不是一定要用的"）：
    只是删减根治失败时的备选方案。先看 v13 复验扩样本 H3——删减被治、聚集消失则不加（默认情形）；
    仅当全臂仍超标才单独评估（复用 Phase 1 机制，`conditioned_sample` 透传 bias_callback +
    `--structure_filter_strength` 开关，⚠️ strength 过强拉偏 H2/降多样性，副作用超限就放弃补丁）。
  ③ 训练侧 C 组件始终保留；bias 补丁只作第二道防线。

## 五、后台任务

| 任务 | 内容 | 状态 |
|------|------|------|
| bteew2epu | v13 配体重训（30ep GPU6）| ⏳ 预解析中（SASA 4957 域 ~1.2h）→ 训练 ~16h |

**复现/监控**：
```bash
tail -f log/v13_ligand_train.log
# 进度
python -c "import json;print(json.load(open('log/train_progress.json')))"
```

# 会话记录 — v13 复验链执行 + H1 双链假异常发现 + 决策 D 停止配体迁移（2026-09-02）

> **状态**：v13 复验链全部完成，**用户决策 D：停止配体迁移**。
> **报告**：`analysis/report/2026-09-02_v13_ligand_validation.md`
> **计划**：`index/PROJECT_LOCAL_V12_2.md §7.8`

## 一、复验链执行时间线

v13 训练（2026-09-01 16.5h）完成后，`run_v13_ligand_validation.sh` 自动推进复验链。

| 环节 | 结果 |
|------|------|
| ① 原始诊断（未校准） | valid slope ~1.30（响应增益）|
| ② 建校准表 | global slope 1.296 / per-protein 17 |
| ③ 泛化采样 n50 | 10 蛋白 × 5 臂 × 50 条（带校准 `--calibrate auto`）|
| ④ 组成分析 | 8/10 蛋白仍删减 0.55-0.69×（仅 1C6O 1.04 达标、1A65 1.35 超上界）|
| ⑤ ESMFold 回折 + TM | 50/50 臂完成 |
| ⑥ 统计 | 修复 `--root` bug 后 H2 70% / H1 35/50 |
| ⑦ PROPKA | 1C6O_n8 PASS（1.27<2.0）|
| ⑧ H3 全臂 | 45/50（1A65 4 臂失败）|
| ⑨ Tm/Sol | S2 **17/50 恶化**（v12.2 配体 9/50）|

## 二、关键发现

### 1. 校准机制完美（校准后 slope 1.00±0.04）
补充跑了校准后诊断（GPU6，与回折 GPU4 并行）——`v10_diag_response_curve.py --calibrate auto`。
valid 10 蛋白 slope 0.92-1.07（均值 1.00±0.04），trainish 1.00±0.01。
未校准 ~1.30 → 校准后 1.00。配体模式 per-protein 校准可靠。

### 2. H1 三个蛋白"折叠失败" = 同源二聚体数据缺陷（从 v9 潜伏）
- 1C6O（89×2）/1AXW（265×2）/1AG0（128×2）为同源二聚体，native 序列 = 两单体串联
- ESMFold 单链折叠串联序列无法重现二聚体 → 连 native 回折 TM 恒 0.5046
- **v9/v10_mompnn/v12_1 蛋白模式 1C6O native TM 均恒定 0.5046**——所有版本一致
- 单体层面验证：fold native 前 89 vs 原 A 链 TM 0.078 → ESMFold 对串联序列折叠与单体构象也不同
- 结论：**三个双链蛋白的 H1 应排除**（验证集数据缺陷，与模型无关）

### 3. write_ref_skeleton resSeq 重复 bug（已修，对 TM 无效但保留）
- 原 PDB 有重复 resSeq（1C6O resSeq 10/11 各重复 2 次）→ US-align 按 resSeq 对齐错位
- 改为 resSeq 连续 1..L（`validate_generalization.py`），10/10 ref 唯一性校验通过
- 对 TM 无影响（双链问题是坐标本质差异）；消除真实 resSeq 重复，下游（H3/LigandMPNN parse）更安全

### 4. ⑨ Tm 链接 + 汇总 bug（已修）
- 链脚本符号链接相对路径 → temberture 找不到 1A65/arm_n2 批量崩
- 改绝对路径重建 50/50 链接，重跑 arm Tm 完成
- 汇总脚本 `v12_2_ligand_tm_sol_summarize.py` 最后参考基线打印 None 值 `:7.2f` 崩溃 → 修 `f_or_dash`

### 5. ⑥ generalization_stats.py `--root` bug（已修）
- 链脚本传 `--root "$OUT/ligand"`，脚本遍历 `root/{mode}/*/validation.json` 需 `--root "$OUT"`
- 改 `--root "$OUT"` 后 H2/H1 统计正常输出

## 三、核心结论：v13（A1+A2）未达标

A1 `pocket_count_loss` 只护 pocket（<8Å），**非口袋 surface 仍删**（frac_floor 0.5）→
删减未根治 → Tm/Sol 反而恶化（17/50 vs 9/50，uncond 基线一致非假象）。
1AS2 全臂新增 TmΔu−6~−7（pocket 保护促使删减集中 surface）、1A65 过度添加 sol 恶化。
H2 70%（差 1 臂）、H3 45/50 与删减同源。

## 四、决策 D（用户 2026-09-02）：停止配体迁移

用户从 A（扩 A1 全 surface）/ B（frac_floor）/ C（组合）/ D（停迁移）选 **D**：
- **v12.2 蛋白模式 = 当前最优交付物**（slope 1.00 / H2 72% / Tm-Sol 0/50）
- 配体模式删减 = **已知局限**，论文如实报告 + `define_pocket.py` fix 缓解
- **不再重训**；A1+A2 代码保留（重启须扩全 surface）
- 配体保守边界：正电+8 / 负电−5 / 长序列检查

## 五、产物

- 校准表 `output/charge_calibration_v13_ligand.json`
- 校准后诊断 `output/v13_ligand_diag_calibrated.json`
- 泛化 `output/generalization_ligand_v13/`（10×5×50 + folds + tm.csv）
- 统计/组成/H3/PROPKA：`output/v13_ligand_gen_stats.json` / `v13_ligand_comp.json` / `h3_ligand_v13.json` / `propka_v13_ligand/`
- Tm/Sol `output/tm_sol_ligand_v13/tm_sol_summary.json`（S2 17/50）
- 报告 `analysis/report/2026-09-02_v13_ligand_validation.md`

## 六、2026-09-02 问答补充（数据统计 + 机制澄清 + AF3 数据准备）

用户追问 5 问，结论已并入报告 §八，要点：
1. **配体 vs 蛋白为何表现差距**：配体线多"删减捷径"（深口袋 frac_sasa 监督盲区 × 配体疏水先验
   × 微调放大 × 成对删逃逸）；蛋白线 MoMPNN 无此通道。
2. **ESMFold 只折单链**，无法验证二聚体/配体结合 → AF3 数据清单
   `analysis/report/2026-09-02_ligand_af3_fold_data.md`（产物路径 + 链拆分 + 判读指标）。
3. **数据统计**：配体训练 4957 域（L 20-500，>400 占 19%）长度覆盖够；CATH 蛋白训练 6710 域
   （median 135，>400 仅 2%）→ 长蛋白对蛋白模式是长度 OOD；验证集 3 个二聚体与训练集单链
   类型不统一。
4. **H2 失败共因**：蛋白/配体两模式共同失败 = 1BJ4 全 5 臂（长蛋白，与配体无关）；2FEO 蛋白
   特有失败（校准表拟合）配体反而过；1AS2 配体特有（删减 + RNA）。
5. **frac_sasa 盲区机制**：`surface_composition_loss` 只监督 frac_sasa≥0.25 表面（无 ceil），
   配体深口袋滑出监管 → 蛋白模式无此盲区故修好；A1 全局化 = 计数锚扩到全部温和改残基
   （几何定义、双向 floor0.7/ceil1.3），绕开 frac_sasa 盲区。

## 七、2026-09-02 二次决策：两任务并行（v12.3 蛋白 + 配体 RNA 扩充）

**用户决策**：保留 v12.2，开两任务并行（详见 memory + `PROJECT_LOCAL_V12_2.md`）：
1. **v12.3 蛋白模式**（子 agent，GPU6）：CATH 补长蛋白（S40 34653 域，L>400 2%→~10%）+ 类型比例保持 + 纯蛋白 + 失败回退 v12.2；验证集删二聚体、类型与训练一致
2. **配体 RNA/DNA 扩充**（子 agent，GPU4）：核糖体高分辨结构（4V4T/9RVC 等）拆全部核糖体蛋白 + 15Å RNA（每残基最近 25 配体原子）；去重；A1 全局化（pocket→surface∪pocket 双向，floor 收紧+λ 提高）；重构验证集

**关键发现（本轮）**：
- 🐛 **LigandMPNN 权重 atom_context_num=25，代码 number_of_ligand_atoms 写 16** → 与权重不匹配，统一改 25（validate_generalization.py 已改）
- A1 无效实测（v13 pocket 带电 0.35-0.65×）→ 全局化方向
- RNA 配体可行性确认（prody 非标准 AA 自动进配体 Y；1B23 RNA 解析 OK；RCSB 网络通）
- 数据统计：配体训练 99.7% 小分子/核苷酸类 10.7%/核酸链 0.3%；CATH 蛋白训练 >400 仅 2%

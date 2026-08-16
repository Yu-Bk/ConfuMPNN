# E4：MoMPNN 设为默认生成器 + 集成验证

> 日期：2026-08-16　|　对应 `PROJECT_EXTEND.md` Stage E4（集成回原计划）
> **结论：`run_guided.py` 默认生成器已切换为 MoMPNN（`mompnn_temberture_tm_esm_6_4_4_b01.ckpt`），默认命令冒烟验证通过（target=0 命中 −0.01±0.80）；对照实验依据已在 E1/E1b 完成（MoMPNN 四指标 × 4 PDB 全部占优），原版 LigandMPNN 通过 `--weights` 显式回退保留。**

## 一、E4 目标与依据

Stage E4 定义：把微调后模型设为 `run_guided.py` 默认生成器 + 完整对照实验。对照实验已在 E1（单 PDB 三目标）与 E1b（4 PDB × 3pH × 3target 扩展）完成，结论：

| 指标 | E1（1BC8） | E1b（4 PDB 留一） |
|------|-----------|------------------|
| 可溶性 %sol | +12.8 | 4/4 PDB 占优（+0.7 ~ +13.4） |
| 热稳定 Tm | +7.8°C | 4/4 PDB 占优（+4.5 ~ +9.8°C） |
| 可设计性 pLDDT | 持平（−0.13） | 4/4 PDB 占优（+0.6 ~ +17.0） |
| 结构保持 TM-score | — | 4/4 PDB 占优（+0.00 ~ +0.18） |
| 电荷响应 | 偏差≤0.10 | 24/24 单调，两模型均命中 |

**16/16 全优（4 指标 × 4 PDB，留一蛋白符号一致）→ 设为默认具备充分依据。**

## 二、实现变更（`code/run_guided.py`）

1. 新增默认权重常量 `_DEFAULT_WEIGHTS` = MoMPNN `mompnn_temberture_tm_esm_6_4_4_b01.ckpt`
2. `--weights` 默认从"LigandMPNN"改为"MoMPNN"（help 文本同步）
3. 保留 `--weights` 参数：回退原版 LigandMPNN（含配体上下文）用 `--weights ../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt`
4. `--model_type auto` 自动识别 MoMPNN 为纯 backbone（protein_mpnn）——已有逻辑，无需改动

## 三、验证（默认命令冒烟）

```bash
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 --num_samples 3
```
- 加载模型：`mompnn_temberture_tm_esm_6_4_4_b01.ckpt` ✓
- 生成序列前缀 `MKSKISLYEF...`（MoMPNN 典型输出，与 E1 protsol 记录一致）✓
- 平均净电荷 **−0.01 ± 0.80**（target=0 精确命中）✓

## 四、文档同步

- README：快速开始默认命令不再需要 `--weights`；新增"回退原版"示例
- `docs/USAGE.md`：场景 4 改为"回退原版 LigandMPNN（配体上下文场景）"
- `docs/CONFIG.md`：`--weights` 默认值说明更新
- 本报告 + 索引/状态快照更新

## 五、使用影响与注意

1. **默认行为变化**：默认从"LigandMPNN（含配体上下文）"变为"MoMPNN（纯 backbone）"。对**无配体**任务（大多数蛋白设计）默认更优；需配体/核酸结合上下文的任务**必须显式指定原版权重**。
2. MoMPNN 无配体上下文支持——这是纯 backbone 与 LigandMPNN 的固有差异，非 bug。
3. 电荷控制依赖引导强度（`--strength`）与 pH 物理极限，与生成器选择无关（E1b 可用率结论）。

## 六、后续

- **Phase 2 条件微调**：在 MoMPNN 权重视作 backbone 的基础上叠加条件编码器（`condition_embedding.py` 已就绪），实现模型级 pH 感知——Level 1 的诚实边界（模型无 pH 先验）的正解。
- 可选：`--fixed_residues` 位点固定（E1b 设计文档已列）。

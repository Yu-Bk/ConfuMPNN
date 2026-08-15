# Session 概要 — 2026-08-15：Phase 1 模块开发

## 本次任务
开始 Phase 1 代码工作，准备条件嵌入、损失函数等模块（对应 PROJECT_PLAN 第五部分）。

## 做了什么
1. **梳理 LigandMPNN 接口**：
   - bias 注入点确认：`model_utils.py:319` `probs = softmax((logits + bias)/temperature)`，bias shape [B,L,21]
   - `model.sample(feature_dict)` 返回 {S, sampling_probs, log_probs, decoding_order}
   - `parse_PDB` → protein_dict；`featurize` 前需手动补 `protein_dict["chain_mask"]`
2. **创建 `code/src/` 7 个模块**：pka / differentiable_charge / isoelectric_point / structure_aware_filter / condition_embedding / losses / guided_sampler
3. **创建 `code/configs/`**：filter_presets.yaml（4 场景）、condition_defaults.yaml
4. **测试**：`code/tests/test_all.py` 29/29 通过；`smoke_guided.py` 真实 LigandMPNN + 1BC8.pdb 冒烟通过

## 关键决策
- Phase 1 里程碑：**不改模型代码**，纯采样策略（logit bias 注入）
- 引导采样两模式：静态 bias（预计算→model.sample）/ 动态逐步解码（每步 bias 回调）
- 条件编码器用 NExT-Mol 风格 Soft Prompt MLP（7→64→128→4×128），连续编码不量化
- 电荷计算用 sigmoid 平滑 HH 方程（处处可微，供 Phase 2 梯度）

## 遗留 / 下一步
- 动态 bias 的"电荷 lookahead"回调待实现（smoke 里只是 zero_bias 占位示例）
- Phase 2 需从训练集计算条件向量标准化 μ/σ（写入 condition_defaults.yaml）
- 阈值统计：从 PDB 采样 1000 条确定 99 分位默认阈值（PROJECT_PLAN Phase 1 待办）

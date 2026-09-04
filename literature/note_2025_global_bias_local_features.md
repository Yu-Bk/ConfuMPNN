# 论文观点记录：global biases 无法处理局部关键序列特征（2025 MPNN bias-redesign 工作）

> 记录于 2026-09-04（ConfuMPNN 论文写作引用备查）。文献细节（作者/期刊/DOI）待补，标注位置 `[CITE_PENDING]`。

## 观点（原意转述）
一篇 **2025 年关于 MPNN 序列设计 bias 重设计**的工作明确指出：

> **Global biases cannot address local sequence features that are critical for binding and developability.**

举例（用户补充的机制性解读）：**无差别地全局抑制表面电荷（global bias）会连带破坏对结合/稳定性关键的局部特征**——如带电残基参与的氢键、盐桥、极性接触等局部分子间/分子内相互作用。

## 为什么与我们相关（ConfuMPNN）
- 我们的 pH-感知电荷条件化在推理侧用 **logit bias 全局推/拉带电残基**（surface 电荷全局增减）——正是"global bias"的实例。
- 实证对应：v12.2/v13/v14 配体系列普遍存在**组成删减捷径**（生成时系统性删带电残基 0.43-0.69× 来凑净电荷，`2026-09-04_v14_clean_validation.md` ④、`2026-09-02_v13_ligand_validation.md` ④）。这种全局"去表面电荷"式删减很可能正是以牺牲结合口袋/表面带电残基的**局部氢键/盐桥网络**为代价。
- 论文论点支撑：**需要位置解析 / 结合位点感知（binding-aware / local）的监督或解码**，而非仅全局电荷约束——对应我们提出的"结合残基 fix / 局部保护"方向（计划见 `session/2026-09-04_fix_binding_localize_plan.md`）。

## 引用状态
- [ ] 定位 2025 原文献（标题/作者/DOI），确认确切表述并核对是否已有对应引用
- [ ] 若命中本仓库 `literature/` 已有条目，替换本文件为指针

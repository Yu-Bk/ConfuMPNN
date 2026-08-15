# P4 — CAPE-MPNN 创新点

> 本文核心创新：**首次**把「去免疫（MHC-I）目标」直接融入 ProteinMPNN 训练，用 LLM 的 DPO 对齐技术生成低免疫可见性的序列，同时保持结构。

## 创新方式（思路层）

1. **首次将去免疫整合进结构驱动的序列生成模型**：此前去免疫工作（Choi/Yachnin/Zinsli 等）多基于规则或传统方法，ML 整合（Bootwala/Gasser/Lyu）都是后处理或预测层面，本文是**首个**把 MHC-I 递呈目标直接并入 ProteinMPNN 生成过程的工作。

2. **DPO 从 LLM 对齐迁移到蛋白去免疫的「类比」**：类比 chatbot「避免使用某些词」↔ 蛋白设计「避免生成被递呈的 k-mer（表位）」。用 MHC-I 递呈预测器扮演「人类反馈」的角色，ProteinMPNN 扮演 foundation model。

3. **明确的 trade-off 分析框架**：把问题定位为「可见性降低 vs 序列质量/结构保真」的双目标权衡，而非只求单边最优。

## 创新模块（实现层）

1. **PWM 递呈预测器（训练加速）**：
   - 用 300 万随机 8–10mer + netMHCpan 标注，训练单 PWM（6 等位基因合并）。
   - 用 PWM 近似替代昂贵的 netMHCpan 在线调用（仅训练用），评估仍用 netMHCpan 4.1。
   - 阈值校准：使 PWM 在随机集上的递呈比例与 netMHCpan 一致。

2. **免疫可见性定义（Definition 1）**：去重计数序列中所有被递呈 8–10mer 的数量；引入相对可见性（÷ 模板可见性）便于跨蛋白比较。

3. **可替换权重（drop-in）**：CAPE-MPNN 与 ProteinMPNN 同架构，权重可直接替换，无缝接入 RFdiffusion/FoldingDiff → ProteinMPNN 等现有工作流。

4. **工程封装**：Docker 容器化（需 GPU + localcolabfold + DE-STRESS/Rosetta），超参搜索脚本化（`cape-mpnn.py` + yaml 配置）。

## 关键数据（创新的有效性）

- DPO 有效降低 MHC-I 可见性，且不损害结构完整性（ColabFold 预测验证）。
- 给出「可见性降低量 vs 序列质量」的 trade-off 曲线，用于选合适 β 等超参数。

## 与其他论文的关联

- 与 P2/P3 同属「DPO 调 ProteinMPNN」的技术路线，但目标是**免疫学性质**（去免疫）而非可设计性/可开发性。
- 四篇中最「专用」的一篇：只针对 MHC-I 一个目标，采样/训练/评估管线为免疫学定制（PWM、netMHCpan、假设患者等位基因集）。
- 与 P1 的共同点：都要在「改序列」的同时「保留功能/结构」，P1 用固定位点，P4 用 DPO + 结构保真评估。
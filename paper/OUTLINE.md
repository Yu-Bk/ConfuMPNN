# ConfuMPNN — 论文框架 / OUTLINE（canonical，2026-09-06 建）

> **本文件 = 论文框架的唯一正本**。图与文章板块映射的轻量版在 `figure/plan_01.md` §I/§J；
> v3 方法学方案在 `index/PROJECT_LOCAL.md`；模型定义/术语见 `analysis/report/2026-09-04_paper_subconclusions.md`。
> 每节给「claim → 证据(报告/数据) → 图 → 草稿状态」。数值出表按 `analysis/report/2026-09-06_number_reporting.md`。

## 0 工作标题
pH 感知的电荷条件化逆折叠：在配体感知结构模型上实现可控净电荷与 pH 响应，及其组成-物理边界。

## 1 Introduction / Related（待写）
- 背景：蛋白工程调 pI/表面电荷 → 可溶/稳定/结合；逆折叠按结构换序列。
- 现状：逆折叠(MoMPNN/LigandMPNN/ESM-IF)可保结构但**无法按目标电荷/pH 控制**；纯序列/生成模型无结构。
- 创新点（claim 候选，待收敛）：
  1. 首个在配体原子上下文结构逆折叠上做 **pH+净电荷双条件**（ConditionEncoder soft-prompt，跨注意力注入冻结 backbone）；
  2. **系统刻画**条件电荷控制的机制与边界：删减捷径 / bias-vs-encoder / 组成(删减↔膨胀) / pH 非边界但需逐 pH 标定。
- 外证：2025 全局 bias 管不住局部（`literature/note_2025_global_bias_local_features.md`）。

## 2 Method
- 框架：冻结 backbone（MoMPNN 蛋白 / LigandMPNN 配体）+ ConditionEncoder((pH,charge)→soft prompt)+复合损失(CE+λ_c 电荷偏差+λ_kl+λ_keep+v12 组成/λ_target+A1)+校准三口径。
- 关键实现文件指针：`code/train_finetune.py`、`src/condition_embedding.py`、`src/conditioned_sampler.py`、`run_guided.py`。
- 数据：CATH/蛋白 + 配体 5371（RNA/DNA 414,7.7%）；验证集 15% 外部 805；测试 in-10。

## 3 Results（claim 逐条，标 evidence/图/状态）
- R1 电荷可控（主结果）：
  - v14 配体 clean：H2 45/50=90% [78.6,95.7]；H1 折叠 50/50；H3 50/50；S2 0/50 —— `2026-09-04_v14_clean_validation.md`；Fig10-14。
  - 蛋白 v12.2：完整链达标（slope1.00，per-protein H2 72%、小样本 74%、S2 0/50）—— `2026-08-31_v12_2_summary.md`。
- R2 相对裸 backbone 的增益集中在极端电荷臂、近 native 靠裸分布 —— exp1 v1+v2（三类均衡）—— Fig I-1/7。
- R3 机制（关键）：bias(逐位贪心) 逐序列命中 ≫ encoder，但 encoder=同号对称删减、bias=同号对称膨胀、对侧置换极少 → 需组成保真/置换 —— exp2+exp2b —— Fig I-2/I-3。
- R4 pH：非边界；逐 pH 现场标定后蛋白 87%/配体 88% 可达；7.4 外推才是假边界；极端 pH 抽查 H1/H3/H4 不坏 —— exp7b —— Fig I-4。
- R5 删减局限表征：三区/温和档也删/极端不对称/fix 不转移/大样本 40% 臂零 —— Task1/2/3 —— Fig15-18。
- R6 v13-vs-v14（数据扩充收益）：H2 64→90、RNA/DNA OOD 消除；但 v14 删得更深 —— 同协议对照 —— Fig19-21。
- R7 应用：E. coli 70S 拆 46 天然核糖体蛋白 v14 可设计性中等偏弱(56%)、强正过冲 —— `2026-09-05_7k00_ribosome_design.md` —— Fig22。

## 4 Discussion
- 全局 bias 局限（2025 外证 + 删减/膨胀双扭曲）→ 需要 local/组成保真。
- encoder 差异化价值（learned 全局 pH 条件，bias 无）→ 但电荷命中弱 → "encoder+bias+置换监督"下一步。
- 校准：表内 per-protein / 表外现场标定 / 逐 pH 标定 —— 三口径使用规范。

## 5 Limitation
- 组成删减未愈（0.43-0.69×，跨版本）；逐序列弥散(std4-6)；bias 膨胀物理代价未测(Tm/Sol/H1 待补)；小样本标定依赖；蛋白模式测试集一度缺碱性单体（已补 1LYZ）。

## 6 待办（论文产出）
- [ ] 图：figure/plan_01.md 全部 Fig（大部分待画；已出 exp7b pH png）
- [ ] 均值/命中率表格按规范出（CI 已备）
- [ ] 每节草稿文字（本 outline 反链到 evidence）
- [ ] 湿实验/表达验证（数据未上传，Fig24）
- [ ] AF3 vs ESMFold（Fig27，可选）

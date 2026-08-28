# 会话记录 — 2026-08-28：同行查新（novelty audit）与 PROJECT_LOCAL §11 更新

> 落盘：2026-08-28｜对应产出：更新 `index/PROJECT_LOCAL.md`（新增 §11 同行查新，原 §11→§12）
> 触发：用户要求核实 "DynamicMPNN 是否已把电荷融入"，并做系统性查新判断是否有同行冲突。

---

## 一、背景与出发点

1. 用户此前质疑"DynamicMPNN 好像已把电荷融入进去"，要求查新确认 ConfuMPNN 是否有创新处。
2. 已用全文阅读核实 DynamicMPNN（`Abrudan et al. 2026, ICLR`）：22 页正文 **0 处 charge / pH / electrostatics / pKa / protonation**；其条件为**构象集合（multi-state backbone）**，"chemical environments"指配体/伙伴原子上下文（与 LigandMPNN 同源的 atom context），**不含电荷**。
3. 全 Zotero 库（588 条非附件条目）扫描：无任何"pH/电荷条件化蛋白序列生成/逆折叠"文献。
4. 据此发起全网查新（2023-2026，扩展 2021-2026）。

## 二、查新执行

- 技能：`nature-academic-search`（MCP 学术源本会话未挂载 → 按技能 fallback 协议使用 `scripts/academic_search.py`，**OpenAlex** 覆盖 CrossRef/PubMed/arXiv/bioRxiv；T1 跨学科 → T2 预印本）。
- 规模：**3 轮 30 组关键词**（含 `pH-conditioned protein design/generation`、`isoelectric point conditioned`、`net charge … generation`、`charge-conditioned/diffusion`、`pH-aware / pI-aware`、`controllable inverse folding`、`property-conditioned ProteinMPNN`、`electrostatic × inverse folding / generation / design`、`pH × ProteinMPNN`、`charge × inverse folding`、2024-2026 最新检索等），每轮 OpenAlex limit 12-15、year-from 2021-2024。
- 命中去重后人工筛读约 280 篇（含全部标题/年份/引用/期刊/摘要）。
- 技术备注：Windows 控制台 GBK 编码导致首轮脚本打印作者名报错，已用 `PYTHONIOENCODING=utf-8` + 结果写 JSON 文件解决。
- 局限：OpenAlex 对极新 2026 预印本可能有滞后；无 Google Scholar / 中文库覆盖。

## 三、查新结论（写入 PROJECT_LOCAL §11.1）

- **未发现直接冲突**：`pH-conditioned protein generation`、`charge-conditioned sequence diffusion protein`、`pI-net-charge conditioned generation` 等精确组合均 0 命中。
- 无任何工作以"连续 pH/净电荷为显式条件 + 结构逆折叠（LigandMPNN/ProteinMPNN 系）+ 可微 HH 电荷目标"做序列生成。
- ConfuMPNN 核心组合在本检索范围内**未被占据**。

## 四、最近邻清单与区分（写入 §11.2，逐条核对过摘要）

1. **SurfPro**（2024 arXiv 2405.06693）：表面+表面生化性质条件化生成（层级表面编码器+自回归解码）→ 条件=表面几何/生化性质场，非 pH/净电荷标量；非 LigandMPNN 微调。
2. **SurfDesign / SurfFold / BC-Design**（2025-2026）：表面/生化感知逆折叠 → 输入特征改进；**BC-Design 的 "biochemistry-aware" 是否含电荷特征待精读确定**。
3. **Controllable protein design with language models**（NMI 2022）：属性标签控制蛋白 LM → 序列空间+离散标签。
4. **Regression Transformer**（NMI 2023）：连续性质回归=条件序列生成 → 序列空间范式，非结构逆折叠。
5. **Integrative multiobjective (NSGA-II)**（PLoS CB 2024）：演化多目标优化做序列设计 → 优化/筛选框架。
6. **LaMBO-2 / Guided Discrete Diffusion**（2023）：离散扩散+性质引导 → 纯序列扩散（已是项目 baseline 候选）。
7. **Electrostatics as a Guiding Principle（酶设计）**（JCTC 2024）：静电计算引导酶设计 → 物理能量法，非学习型条件。
8. **pH-responsive filaments**（Nat. Nanotechnol. 2024）、**pH-responsive antibody nanoparticles**（NSMB 2024）：设计 pH 触发组装/解离结构（埋藏 His）→ 任务的**不同**（结构开关 vs 固定工作 pH 的电荷控制）——审稿人最易混淆项。
9. **CamSol pH-dependent solubility**（2023）：pH 依赖可溶性预测器 → 非生成器；采用为验证 oracle。
10. **ABACUS-T / MapDiff / DynamicMPNN**（2024-2026）：多模态/多构象条件逆折叠 → 条件均为结构类，无理化标量。

## 五、决策与定位调整（写入 §11.3-11.4）

1. **claim 措辞**："首个 pH/电荷条件化逆折叠"必须加限定：*首个在配体感知结构逆折叠上，以连续 pH+净电荷为显式条件并用可微 HH 电荷目标训练的方法*。
2. **护城河**：机制发现（删减捷径→电荷斑块丢失→表面疏水化）+ v10 表面可控制电荷设计；查新未发现同类机制研究。
3. **Related Work 专门段**：区分"pH 触发结构开关（filaments/nanoparticles）"vs"固定工作 pH 的电荷/组成控制"。
4. **新增行动项**：精读 SurfPro / SurfDesign / BC-Design 全文；补 arXiv/bioRxiv 页面级检索；CamSol-pH 接入验证连线（并入 §3.3 可选 oracle）；按 §11.2 表写作 Related Work。
5. PROJECT_LOCAL 更新：新增 §11、原 §11→§12、§3.3 加入 CamSol-pH、§12.1 更新投稿定位注记、尾部日期更新。

## 六、遗留

- 2026 极新预印本（尤其 2026H2）尚未覆盖完全；BC-Design 全文待精读。
- 若后续决定投稿，投稿前需再跑一轮终检（PubMed/Google Scholar + arXiv 订阅），并把最近邻表同步进 manuscript。

*最后更新：2026-08-28。*

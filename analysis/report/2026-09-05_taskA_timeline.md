# Task A 时间线 — 蛋白模式（MoMPNN）与配体模式（LigandMPNN）版本更迭（2026-08→09）

> 配套主报告：`analysis/report/2026-09-05_protein_history_vs_ligand_deletion.md`
> 图例：🟦 蛋白线（MoMPNN backbone）｜🟨 配体线（LigandMPNN backbone）｜🔧 通用机制/修复

```mermaid
timeline
    title ConfuMPNN 版本更迭
    section 2026-08-16~19 阶段成果（v7/v9 定稿）
        08-16..18 : E1 基线 & Phase1-3 迭代
                  : v2→v7 数据/占位符/平衡/课程学习
                  : 🔍 P1 发现：删减捷径
                  : 1BJ4 native 105 带电残基→~18
        08-18/19 : 🟦 v7 = MoMPNN 蛋白编码器（冻结，preview1.0.0）
                  : 🟨 v9 = LigandMPNN 配体编码器（冻结）
                  : 泛化验证 + 电荷边界分层
    section 2026-08-26/27 v3 论文方案 + P0
        08-26/27 : 🔧 PROJECT_LOCAL v3（A 解耦+B 表面加电荷监督+C 结构惩罚）
                  : 🔧 P0 代码：target 自动补全/RMSD/PROPKA/fractional SASA/pH 自适应过滤
    section 2026-08-27~29 v10/v11 消融（蛋白线）
        08-27/28 : 🟦 v10 训练（A+B+C 全开）
                  : ❌ 泛化退化：负向过冲，n8 命中 0
        08-28 : 🔧 17 蛋白响应诊断 → 推翻"外推"假说
               : 根因 = B(L_add)与删减捷径"双算" + decouple 弱化 native 锚
        08-29 : 🟦 v11a(B-OFF)/v11b(A-fix)/v11c(全fix) 消融
               : 结论：B 纯有害应弃；A-fix/C-fix 无效
               : 四版 slope 1.41~1.67 全未达标
        08-29晚 : 🔧 bias 排查 → 根因=编码器响应增益
                 : 线性校准零重训修复 slope→0.88~0.95
    section 2026-08-29~31 v12 家族（蛋白线）
        08-29深夜 : 🟦 v12 训练（组成双计数+GRAVY+surface charge target）
                  : ❌ 治删成功但过度添加 1.5-2×（slope 1.85）
        08-30 : 🟦 v12.1 调参（floor 0.5/margin 0.4/λ 0.2）
               : ✅ 组成健康（治过度添加）；slope 校准后 1.04
        08-31 : 🟦 v12.2 = v12.1 + λ_target 0.2（表面电荷锚）
               : ✅ 蛋白线当前最优交付：slope 1.00 / H2 72% / 小样本标定 74%
               : ✅ Tm/Sol S2 0/50；hold-out 40.6%（global 兜底固有上限）
    section 2026-09-01~02 配体迁移与失败（配体线）
        09-01 : 🟨 v12.2 配体重训（16.5h）
               : ⚠️ H2 72% 达标但组成 8/10 删 0.53-0.65×（定向口袋）
               : 🔧 4 层证据根因=监督逃逸×疏水先验×微调放大
        09-01/02 : 🟨 v13 重训（A1 pocket_count + A2 三块互斥分区）
                  : ❌ 未达标：surface 仍删 8/10 0.55-0.69×；Tm/Sol 恶化 17/50
        09-02 : 🔧 决策 D 停配体迁移（后用户重启）
    section 2026-09-02~04 双轨：v12.3 蛋白长域 + v14 配体 RNA/DNA
        09-02 : 🟦 v12.3 启动：+455 长 CATH 域重训（7165 域）
               : 🟨 v14 启动：RNA/DNA 扩充 414 域 + atom_context 16→25 + A1-global
        09-03 : 🟦 v12.3 完成：in-5 H2 退步（small 92%→80%）
               : ⚠️ 组成删减系统性加重（全 <1；1A65 1.2→0.86/0.73）
        09-04 : 🟨 v14 clean（in-10）：H2 90% / H1 100% / S2 0/50
               : ❌ 组成删减未根治 0.43-0.69×（全 10 蛋白）
               : 🔧 删除定位/fixbinding/largen 三任务诊断
               : fixbinding：口袋保 100%，surface/core 删除不动 → 删除是全局默认
               : largen：n200 三达标序列仅 5.2%，删除为第一瓶颈
    section 2026-09-05 Task A
        09-05 : 📊 本文档：蛋白史 + 蛋白/配体删减差异机制分析
```

## 一句话版本线
- 🟦 **蛋白线（MoMPNN）**：v7（无组成监督，P1 删减 105→18）→ v10/v11（A+B+C 失败与消融，B 弃）→ v12（组成 floor，过度添加）→ v12.1（调参）→ **v12.2（+λ_target 表面电荷锚，最优）** → v12.3（加长域，覆盖内退步未采纳）。
- 🟨 **配体线（LigandMPNN）**：v9（旧边界）→ v12.2-ligand（删减 0.53-0.65×）→ v13（A1 护口袋，surface 仍删）→ **v14（RNA/DNA + A1-global，删减 0.43-0.69× 仍未根治）**。
- 🔧 **贯穿机制**：删减捷径（成对删 D+K 保净电荷）自 v7 即存在；v12 起的"表面 floor + GRAVY + λ_target"把蛋白模式压到"轻删/均匀"，配体模式因深口袋盲区×疏水先验仍在逃逸。

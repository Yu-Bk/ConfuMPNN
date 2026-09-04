# Task3 大样本"三达标"序列搜索（v14 ligand in-10, n=200/臂）

日期：2026-09-04 ｜ GPU：cuda:4（共享） ｜ 输入：validation_manifest_v14_in.json (10 蛋白) × 5 臂 × n=200

## 0. 一句话结论

**不是"模型从不生成合格序列"：n=200 下有 30/50 臂 (60%) 存在"三达标"序列，9/10 蛋白至少一臂存在；整体三达标率 5.2% (523/10000)。** 但 20/50 臂 (40%) 即使采样到 n=200 仍为零——主要集中在删减严重的 1BJ4 (全 5 臂零) 及 1AS2/2FEO/5CQH 等多数臂。把 n 从 50 放大到 200，能把"存在三达标"的臂从 24 增至 30，把三达标序列总数从 127 增至 523，**但救不回删减系统性的 20 个臂**。未达标主因：**判据②删减**（整体通过率 17.5%）≫ 判据①电荷达标（32.1%）≫ 判据③ H3（99.9%）。H3 聚集几乎不构成限制。


## 1. 方法 / 口径（简）

- 采样与 validate_generalization 对齐：target = round(native_charge@pH7.4) + Δ（native/n2/p2/n8/p8 = 0/−2/+2/−8/+8）；per-protein 电荷校准表 charge_calibration_v14_ligand_clean.json（tgt_eff=(tgt−b)/a）；conditioned_sample 注入，temperature=0.3，seed=2000+k；ligand 模式 atom25。

- 三达标（逐序列）：① dev = |net_charge − target| ≤ 2；② del_ratio = (D/E+K/R)/(native D/E+K/R) ≥ 0.7；③ H3：结构感知 4 规则 union 违规率 ≤ native_ref 违规率 + 0.05（与 H3 判据同口径的 per-seq 版）。

- 输出：output/largen_v14/<pdb>_arm_<arm>/{seqs.fa, stats.json}、<pdb>_summary.json、summary.json；汇总 output/largen_v14_summary.json。


## 2. 存在性总览（n=200/臂）

- 50 臂中 **30 臂 (60.0%)** 至少存在 1 条三达标序列；**9/10 蛋白**至少一臂存在。唯一全零蛋白 = **1BJ4**（L=470，最长；所有臂主因删减）。

- 按臂方向（存在臂数 / 10）：native 5，n2 7，p2 5，n8 7，p8 6。

- 10000 条逐序列通过率：电荷 32.1% (3206)，删减 **17.5% (1749)**，H3 99.9% (9985)，**三达标 5.2% (523)**。

- 主因（判据③ H3 通过率 ~100%，几乎不卡；删减是全局最弱一环）。


## 3. 逐蛋白×臂明细

`C/D/H pass` = 200 条中分别满足电荷/删减/H3 的条数；`triple` = 三者同时满足条数；`p10..p200` = 前缀累计三达标条数（=存在率随 n 的曲线）；`del_m` = 删减倍率均值；主因列仅当 triple=0 时标"若零→主因"，triple>0 时标剩余未达标的主因（供参考）。


| pdb | cat | L | native Q | arm | target | C pass | D pass | H pass | triple | rate | p10 | p25 | p50 | p100 | p200 | del_m | dev_m | 主因 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6D2O | small_mol | 209 | -6.2 | native | -6 | 69 | 29 | 200 | 6 | 0.030 | 0 | 1 | 1 | 3 | 6 | 0.57 | 3.80 | deletion |
| 6D2O | small_mol | 209 | -6.2 | n2 | -8 | 55 | 44 | 200 | 9 | 0.045 | 1 | 1 | 2 | 6 | 9 | 0.61 | 4.00 | deletion |
| 6D2O | small_mol | 209 | -6.2 | p2 | -4 | 55 | 21 | 200 | 3 | 0.015 | 0 | 1 | 1 | 1 | 3 | 0.54 | 3.85 | deletion |
| 6D2O | small_mol | 209 | -6.2 | n8 | -14 | 63 | 152 | 200 | 46 | 0.230 | 3 | 5 | 10 | 22 | 46 | 0.80 | 4.51 | charge |
| 6D2O | small_mol | 209 | -6.2 | p8 | +2 | 72 | 11 | 200 | 3 | 0.015 | 0 | 1 | 1 | 1 | 3 | 0.53 | 3.58 | deletion |
| 1AS2 | nucleotide | 312 | -2.7 | native | -3 | 65 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.46 | 4.02 | deletion |
| 1AS2 | nucleotide | 312 | -2.7 | n2 | -5 | 67 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.48 | 3.88 | deletion |
| 1AS2 | nucleotide | 312 | -2.7 | p2 | -1 | 67 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.45 | 3.96 | deletion |
| 1AS2 | nucleotide | 312 | -2.7 | n8 | -11 | 63 | 5 | 200 | 2 | 0.010 | 0 | 0 | 1 | 2 | 2 | 0.58 | 3.86 | deletion |
| 1AS2 | nucleotide | 312 | -2.7 | p8 | +5 | 66 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.48 | 4.48 | deletion |
| 2FEO | nucleotide | 221 | -6.9 | native | -7 | 59 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.47 | 4.20 | deletion |
| 2FEO | nucleotide | 221 | -6.9 | n2 | -9 | 53 | 2 | 200 | 1 | 0.005 | 0 | 0 | 0 | 1 | 1 | 0.50 | 4.20 | deletion |
| 2FEO | nucleotide | 221 | -6.9 | p2 | -5 | 61 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.46 | 4.15 | deletion |
| 2FEO | nucleotide | 221 | -6.9 | n8 | -15 | 49 | 95 | 200 | 19 | 0.095 | 0 | 0 | 1 | 9 | 19 | 0.69 | 4.68 | charge |
| 2FEO | nucleotide | 221 | -6.9 | p8 | +1 | 64 | 29 | 200 | 8 | 0.040 | 1 | 1 | 2 | 6 | 8 | 0.60 | 4.53 | deletion |
| 5CQH | nucleotide | 183 | -5.5 | native | -6 | 88 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.43 | 2.96 | deletion |
| 5CQH | nucleotide | 183 | -5.5 | n2 | -8 | 83 | 4 | 200 | 1 | 0.005 | 0 | 0 | 0 | 0 | 1 | 0.49 | 3.17 | deletion |
| 5CQH | nucleotide | 183 | -5.5 | p2 | -4 | 101 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.40 | 2.64 | deletion |
| 5CQH | nucleotide | 183 | -5.5 | n8 | -14 | 57 | 105 | 200 | 24 | 0.120 | 2 | 5 | 8 | 13 | 24 | 0.72 | 3.96 | charge |
| 5CQH | nucleotide | 183 | -5.5 | p8 | +2 | 92 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.42 | 2.67 | deletion |
| 1CGE | metal | 162 | -11.7 | native | -12 | 79 | 33 | 200 | 14 | 0.070 | 1 | 1 | 1 | 7 | 14 | 0.60 | 2.84 | deletion |
| 1CGE | metal | 162 | -11.7 | n2 | -14 | 84 | 84 | 200 | 38 | 0.190 | 1 | 4 | 9 | 23 | 38 | 0.68 | 2.86 | charge,deletion |
| 1CGE | metal | 162 | -11.7 | p2 | -10 | 95 | 7 | 200 | 1 | 0.005 | 0 | 0 | 0 | 0 | 1 | 0.52 | 2.67 | deletion |
| 1CGE | metal | 162 | -11.7 | n8 | -20 | 66 | 193 | 199 | 64 | 0.320 | 2 | 9 | 17 | 33 | 64 | 0.92 | 3.53 | charge |
| 1CGE | metal | 162 | -11.7 | p8 | -4 | 96 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.35 | 2.49 | deletion |
| 1BJ4 | long | 470 | +0.4 | native | +0 | 47 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.46 | 4.86 | deletion |
| 1BJ4 | long | 470 | +0.4 | n2 | -2 | 51 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.45 | 4.86 | deletion |
| 1BJ4 | long | 470 | +0.4 | p2 | +2 | 48 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.47 | 5.04 | deletion |
| 1BJ4 | long | 470 | +0.4 | n8 | -8 | 46 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.47 | 5.10 | deletion |
| 1BJ4 | long | 470 | +0.4 | p8 | +8 | 50 | 2 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.54 | 5.30 | deletion |
| 21KL_A | RNA | 237 | +10.0 | native | +10 | 48 | 28 | 200 | 8 | 0.040 | 0 | 0 | 2 | 4 | 8 | 0.60 | 4.61 | deletion |
| 21KL_A | RNA | 237 | +10.0 | n2 | +8 | 48 | 14 | 200 | 4 | 0.020 | 0 | 0 | 1 | 2 | 4 | 0.58 | 4.57 | deletion |
| 21KL_A | RNA | 237 | +10.0 | p2 | +12 | 52 | 39 | 200 | 7 | 0.035 | 0 | 0 | 0 | 1 | 7 | 0.63 | 4.51 | deletion |
| 21KL_A | RNA | 237 | +10.0 | n8 | +2 | 42 | 12 | 200 | 1 | 0.005 | 0 | 0 | 0 | 0 | 1 | 0.57 | 4.57 | deletion |
| 21KL_A | RNA | 237 | +10.0 | p8 | +18 | 47 | 153 | 200 | 39 | 0.195 | 4 | 5 | 8 | 15 | 39 | 0.77 | 4.60 | charge |
| 5O60_E | RNA | 209 | +11.2 | native | +11 | 64 | 13 | 200 | 4 | 0.020 | 1 | 1 | 2 | 3 | 4 | 0.55 | 3.74 | deletion |
| 5O60_E | RNA | 209 | +11.2 | n2 | +9 | 61 | 4 | 200 | 1 | 0.005 | 0 | 0 | 0 | 1 | 1 | 0.52 | 3.64 | deletion |
| 5O60_E | RNA | 209 | +11.2 | p2 | +13 | 65 | 32 | 200 | 8 | 0.040 | 1 | 1 | 5 | 6 | 8 | 0.59 | 3.76 | deletion |
| 5O60_E | RNA | 209 | +11.2 | n8 | +3 | 69 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.46 | 3.25 | deletion |
| 5O60_E | RNA | 209 | +11.2 | p8 | +19 | 61 | 152 | 200 | 48 | 0.240 | 4 | 8 | 13 | 25 | 48 | 0.76 | 3.89 | charge |
| 3MXB_A | DNA | 153 | +7.9 | native | +8 | 83 | 75 | 200 | 29 | 0.145 | 3 | 5 | 9 | 16 | 29 | 0.67 | 3.28 | deletion |
| 3MXB_A | DNA | 153 | +7.9 | n2 | +6 | 82 | 33 | 200 | 12 | 0.060 | 1 | 1 | 4 | 8 | 12 | 0.61 | 3.28 | deletion |
| 3MXB_A | DNA | 153 | +7.9 | p2 | +10 | 74 | 129 | 200 | 46 | 0.230 | 1 | 6 | 13 | 27 | 46 | 0.73 | 3.27 | charge |
| 3MXB_A | DNA | 153 | +7.9 | n8 | +0 | 80 | 20 | 200 | 7 | 0.035 | 0 | 0 | 1 | 4 | 7 | 0.56 | 3.15 | deletion |
| 3MXB_A | DNA | 153 | +7.9 | p8 | +16 | 65 | 200 | 186 | 62 | 0.310 | 2 | 6 | 14 | 29 | 62 | 1.03 | 3.74 | charge |
| 9DWG_L | DNA | 323 | +4.0 | native | +4 | 50 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.47 | 4.73 | deletion |
| 9DWG_L | DNA | 323 | +4.0 | n2 | +2 | 56 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.45 | 4.60 | deletion |
| 9DWG_L | DNA | 323 | +4.0 | p2 | +6 | 50 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.50 | 4.86 | deletion |
| 9DWG_L | DNA | 323 | +4.0 | n8 | -4 | 50 | 0 | 200 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0.41 | 4.70 | deletion |
| 9DWG_L | DNA | 323 | +4.0 | p8 | +12 | 48 | 29 | 200 | 8 | 0.040 | 0 | 0 | 1 | 2 | 8 | 0.64 | 5.23 | deletion |

> 注：`triple_rate` 为小数（1CGE n8=0.320 即 32.0%）。主因列对 triple=0 的臂即"若零主因"；对有 triple 的臂为"剩余未达标序列的主因"（同脚本 main_cause_if_none 字段）。


## 4. 存在率随 n 曲线（多采样是否可救）

| 前缀 n（每臂取前 n 条） | 累计三达标序列总数（跨 50 臂） | 首次出现三达标的臂数 | 至少已有三达标的臂数 |
|---|---|---|---|
| 10 | 28 | 15 | 15 |
| 25 | 62 | 3 | 18 |
| 50 | 127 | 6 | 24 |
| 100 | 270 | 3 | 27 |
| 200 | 523 | 3 | 30 |

解读：
- n 从 50 → 200：存在三达标的臂数 24 → 30（新增 6 个"稀有"臂），累计合格序列 127 → 523（约 4.1×）。
- 但 20 臂 (40%) 到 n=200 仍为零 → 对这些臂不是采样量问题，是模型删减的系统性失败（见 §5）。
- 即便在"有救"的臂里，三达标率也低（多数 <5%，最高的 1CGE n8=32%、3MXB_A p8=31%、5O60_E p8=24%）。

## 5. 未达标主因落在哪条

- 全局 10000 条：删减 17.5% 通过 < 电荷 32.1% 通过 ≪ H3 99.9% 通过。→ **判据②（不重删带电残基，D/E+K/R≥0.7×native）是主瓶颈**。
- triple=0 的 20 臂全部标主因 deletion（模型把这些臂的生成序列带电残基总数压到 native 的 0.41–0.58×，del_m 见表）。
- triple>0 的臂中，主因呈现方向依赖：n8/近 native 臂多为 **charge**（均值电荷在 target 附近但逐序列散布 σ≈3–4，dev≤2 只覆盖 ~30%）；正电臂（21KL_A/5O60_E/3MXB_A p8、3MXB_A p2）模型倾向直接加 K/R → 删减常达标，主因回到电荷散布。
- **H3 聚集几乎从不构成限制**（H pass 9985/10000）。需注意：H3 的 R4 同号簇规则在整链 Cα 8Å 连通下退化为"总带电残基数"口径，native_ref 基线本身 ≈ DK_native/L，故该判据实际在约束"带电残基总数不要超过 native+5pp"，不是真正的空间聚集；逐序列 local 规则（R1+R2+R3）计数近 0（见 stats.json h3_local）。

## 6. Pareto(dev vs deletion) 前沿示例

每条臂的 Pareto 前沿示例（含 dev、del_ratio、是否三达标、完整序列）存于 `output/largen_v14/<pdb>_arm_<arm>/stats.json` → `pareto_examples`；最佳三达标序列在 `best`。下为各蛋白代表性 best 三达标示例：

| pdb arm | k | dev | del_ratio | DK(DE/KR) | 序列 |
|---|---|---|---|---|---|
| 6D2O native | 182 | 0.38 | 0.857 | 36(21/15) | SSQQKQEARQQKQNQVKQEDPNFYNNLNQKEQPQTLVITCDNKGVPPEQLINAKEGNLYVYQNQGCIVNPSNKEVLGVLE… |
| 6D2O n2 | 68 | 0.19 | 0.714 | 30(19/11) | SEQQQIEDNEQLKNDIQQQNPTFFDELNKQSEPSTLAITCNDPRVPPTKLIDAQPGELYIYQNAGHIFLPSDKASLGVLA… |
| 6D2O p2 | 10 | 0.23 | 0.762 | 32(18/14) | SNEAEIQNQQAEQNAIEKKNPAFYKKLNQQQQPQVLAITCNDAAVPPQRLMNYQEGDLYVLSNQAAIFLPNDARSLGVLQ… |
| 6D2O n8 | 137 | 0.18 | 0.714 | 30(22/8) | SNNDREQQRQQQRAAREQENPEFYATLAKDENPQTLAITCDDPATPPSELISARPGELYVYQNAGHIVLPENNDVLEVIA… |
| 6D2O p8 | 10 | 0.34 | 0.714 | 30(14/16) | SNEAEIQRQQAEQNAIEKKNPAFYKKLAQQQQPQVLAITCNNNAVPPQRLMNYQEGDLYILSNQAAIFLPNNARSLGVLQ… |
| 1AS2 n8 | 90 | 0.83 | 0.762 | 64(37/27) | REVTVLILGGPGCGQTTLLLQIIIIYTQGFTQDQRKEYQEVIWQNTLSGVQSLIEAMKQLKVDFGSPECQELAESLWEYT… |
| 2FEO n2 | 60 | 0.07 | 0.754 | 43(26/17) | EQAPAVALDGPPGSGQQQVAQAIAGLYEWRLLETGQIFRALAYAAIIQNIDVSNPGALVELASNLNIKFSEKEEELQVYL… |
| 2FEO n8 | 163 | 0.01 | 0.754 | 43(29/14) | GVAEAVAIDGPPGTGQEEVCQTIAQRFGWTLALIGAIYRALALAALQQNVPVENEGALVTLAQQMEIKFSGEEDTLQVWL… |
| 2FEO p8 | 0 | 0.00 | 0.754 | 43(21/22) | EQAKALAVDGPPGSGQGSLCQQLAQQHRWKLVQLGLIFRALALAATEKKVDVSNPGQLVPLAQQLNVTLSKAERSLTVYL… |
| 5CQH n2 | 174 | 1.84 | 0.744 | 32(19/13) | QVPLLDKRLFYEIFNNDPSLQKRAQTLVLYQVAELENGQEQELPSLQGWITNENGVHAYLNLLAQIPSLNLNPAKRYLIT… |
| 5CQH n8 | 36 | 0.01 | 0.744 | 32(23/9) | EIKLLNPALFYQLFNNNPELQSRANTLLLYKVAQLEDGQAIIESDLQGYIFKQNGVHAYLRFLSQVPKLQLDPSQDYLVT… |
| 1CGE native | 137 | 0.21 | 0.778 | 28(20/8) | TQKAQPTWETLELTYRILNYTNDLPTDQVVQAVQRALQLWSAVTPLQFQEVSSGYADIQIEFVNGNHGDDQPFNGKGGRA… |
| 1CGE n2 | 192 | 0.18 | 0.778 | 28(21/7) | EEEPQPTWEKLNLTYKILNYTRNLPKQQVIKAIQEALDLWAAVTPLTFAEVENGNADIQIAFVSGNHGDNRPFNGPGGQI… |
| 1CGE p2 | 186 | 1.78 | 0.722 | 26(19/7) | AQEEEPTWEKLELTYQILNYTNNLPSNEVVQAVRSALQLWSAVTPLTFEEVQSGFADINIAFVKGDHGDDKPFNGKGGQI… |
| 1CGE n8 | 179 | 0.11 | 0.722 | 26(23/3) | ELEEQPTWENTNLTYQILNFSQNLPEQEVIECVKAAFSLWANVTPLTFEKVDEGLADILIAFVAGDHGDSRPFDGPGGQI… |
| 21KL_A native | 91 | 0.06 | 0.710 | 44(17/27) | QWTKQLSELISQYYELRKEQQRYENRLLSIEQKQRRQRPQEQAQLRKEQQEIQKRLKEIEKNLQQLVKNSPDPVVQSLLG… |
| 21KL_A n2 | 158 | 0.03 | 0.710 | 44(18/26) | ESKKPLQKLLSQYFKLLKESQQYSQQLLSLEEGTSQVDPKTQQELKKKLKQISKKLKKIEKQLEEFVSNHQNPIVQSLLS… |
| 21KL_A p2 | 101 | 0.06 | 0.710 | 44(16/28) | QQQRPLEALLRQHDQLQRQATSYAQQILAFKEQTVKVDKKTVQQLEQQLKRTSQQLKEIDKQLKKYISNSPDPVVQALLS… |
| 21KL_A n8 | 130 | 0.09 | 0.806 | 50(24/26) | QSRTELSELVSQYFRLRKESQKYKKQLLALQENQRPFKPSIAKQLQQQLQQTSSELSEVEKELQKLVSESQNPIVQSLLS… |
| 21KL_A p8 | 192 | 0.02 | 0.806 | 50(16/34) | AERERLQKLLAKQDRLQSQQLSYQTQILSIQEETQKRSPKKVKQLAKKLEKISKELTRVDKELKEFVESHQNPIVQSLLS… |
| 5O60_E native | 43 | 0.05 | 0.726 | 37(13/24) | PIKLQVYTYNGKILHTVILPESVFSVQVNKSLLEQVIQAQKNAARQGSAERQTVGEVTGGGRQPSANRGTGAAQLGSNIQ… |
| 5O60_E n2 | 57 | 0.20 | 0.726 | 37(14/23) | PLEIQVLSYQGSVLTTVLLPKEIFAVKVNEKLITQVIVAQQNSAQQGNAARKTVGEVNGGGQQPAANRGTGKAQWGSTVA… |
| 5O60_E p2 | 198 | 0.06 | 0.726 | 37(12/25) | PIEIKVLSYQGQELHTVVLPTQIFSVKVNETLIEEVIKAQQRAAAQGSAANKTVGEVAGGGRQPYAARGTGQAPLGSTRA… |
| 5O60_E p8 | 197 | 0.02 | 0.765 | 39(10/29) | PIRLQVTSYQGQILTTVILPQRIFATEVNMPLIKRVIKAEAANASQGSAARKTVGEVQGGGKKPAKNRGTGASTVGSSRQ… |
| 3MXB_A native | 12 | 0.08 | 0.737 | 28(10/18) | AKQLPTQFLQELAQLIDQNGSIIAQLQPDPEAQFGYRIELTLFVHQKNRKRARLQRLVQQLGAGYVFQNGNVCQYVLSQQ… |
| 3MXB_A n2 | 6 | 0.13 | 0.737 | 28(11/17) | AQQYPRQFLQQLANEINNNGSIIAQLQPNKEAKFNYQISLVLYVTQNTEQQEELQQLVEQLGAGRVFRNGKVAQFILSQT… |
| 3MXB_A p2 | 66 | 0.02 | 0.737 | 28(9/19) | SPKYNREFLQEFAQELNANGSIIAQLQPNPSAQFGYRIELTLFYTQKTKKQKKLQKLVEQIGAGLVVQNGKVSQFILSQF… |
| 3MXB_A n8 | 186 | 0.01 | 0.789 | 30(15/15) | QQEYNESFLKTLSEEINNNGSIIAQLRPSNDAQFGYALKLSLYITKNAEKQEELEKLVQQLGDGFVFQNGSVAQFVLTQT… |
| 3MXB_A p8 | 58 | 0.02 | 0.895 | 34(9/25) | SKKFKQRFLQQLATQVDNDGSIVAQLQPAKWAQFEYRLRLTLFYTQKTSQKSELQALVQQIGHGYVFKNGKSAQFVLQQK… |
| 9DWG_L p8 | 161 | 0.22 | 0.735 | 72(30/42) | PNRLIVKTLQQLASIAENVDNAPYKAAAYTKAAEQISELPQKVRSGKQVQQQQGVGKSIAQKIDEILSTGYLPDLEKEKK… |

完整 Pareto 前沿示例序列放于各臂 stats.json 文件头（`pareto_examples[].seq`）。

## 7. 对 v14 的判定

科学问题结论：**v14 的删减/电荷失败既不是"从不生成合格序列"，也不是单纯"稀有事件多采样即可救"，而是二者混合**：
1. 合格（三达标）序列确实存在且非零概率生成：整体 5.2%，在 60% 的蛋白×臂组合里 n=200 至少能捞到 ≥1 条，多采样确有增量（50→200：臂 24→30、条数 127→523）。
2. 但概率系统性偏低，主因是删减捷径（D=17.5%），且在 20/50 臂（含最长蛋白 1BJ4 全部臂）n=200 仍为零 → 这些情形下 v14 本质不生成合格序列。
3. H3 聚集判据在本口径下不限制；真正需要修的是"带电残基总数不被删减"（v12.2/v13 已指出的删减捷径），电荷逐序列散布大是第二瓶颈。

---

脚本：`code/tests/ligand_v9/largen_search_v14.py`（采样+指标）、`code/tests/ligand_v9/summarize_largen_v14.py`（汇总）。产物根：`output/largen_v14/`。

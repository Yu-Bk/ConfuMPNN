# v10 修复包（v11 路线）— 2026-08-28（诊断已完成版）

> 一句话：诊断已跑（17 蛋白响应曲线，独立复核一致），**结论已修正**——外推不是主因，
> 是"负向响应增益失控 + B(L_add)与删减捷径双算 + decouple 弱化 native 锚"。
> 下一步 = **先做 B-OFF 消融重训**（最可能一举分清责任），再按实验矩阵推进。

## 1. 诊断结论（2026-08-28，原始 JSON 复核一致）

| 指标 | 数值（我复核） | 结论 |
|---|---|---|
| 全区 slope（trainish/valid）| 1.50 / 1.62 | 训练域与验证域都 >1 |
| **区内 slope（native±12，n=17）** | **1.59 ± 0.57** | 训练覆盖区内响应就失控 → **外推假说未坐实** |
| 负区外 slope（n=13）| 1.59 ± 0.65 | 更负要求→超线性过冲 |
| 正区外 slope（n=9）| **1.10 ± 0.34** | 正值要求→基本正常 |
| native 自洽点 | 负 native 蛋白 dev 12~33 | decouple 弱化 native 锚 |
| 正 native 蛋白（2d3yA00/8etcb01/1l3eB00）| 区内 0.69~1.15 | 模型没坏，坏在"负向响应" |
| 确认 MoMPNN 实际训练 | **A+B+C 三组件全开**（log + 管线脚本）| 报告 §5"只改 decouple"归因错误，已修正 |

**核心机制**：需要更负时，模型走"删 K/R"老捷径（P1 根因），B 又要"表面加 D/E"→ 同向双算
→ 净效果 ≈ 2×Δ；正向因可删 D/E 稀少而有界 → 不放大。decouple 把 30% 样本与 native 解耦 →
自洽 native 点也过冲。深负区外推只是放大器。

## 2. 下一步实验矩阵（每版 30 ep MoMPNN，~2h；单 GPU 按序跑）

> 纪律：**一次只动一个变量**（v10 教训：A+B+C+数据范围耦合无法归因）。

| 版 | 改动（相对 v10 = decouple±12 + B λ0.3 + C boost1.5）| 目的 | 判据 |
|---|---|---|---|
| **v11a = B-OFF（先跑）** | 关掉 L_add（`--add_supervision` 不加），其余与 v10 完全一致 | 检验"B 双算"是否为负向放大主因 | 诊断：负向（区内+区外）slope 是否回落 ≈1；若回落 → B 是主放大器 |
| **v11b = A-fix 单开** | 只换 `--decouple_absolute [-35,20]`，不带 B/C；A7 孤立 decouple 与覆盖 | 检验 decouple/覆盖贡献 | 负向 slope 回落 → decouple 是元凶 |
| **v11c = 全fix** | A-fix + B-fix(λ0.1 + scale 0.5，见补丁) + C-fix(逐样本 boost) | 目标版 | 全区 slope≈1、|截距|<1 |
| v11d = C-OFF（若需要）| 只关 C | 检验 C | 同上 |

**先 v11a**：若 B-OFF 就回落，直接确认 B 为主因 → v11c 用 B-fix 版即可；若 B-OFF 不回落，
再跑 v11b（A-fix）判定 decouple；两者都测后决定 v11c 的组合权重。

判据统一用 `v10_diag_response_curve.py`（同一 manifest + targets + n，换 checkpoint）：
- **通过**：区内 slope 落回 [0.9, 1.15]、正负区外 slope 均 <1.3、|截距|<1。
- **未过**：继续下一个单变量版，或接受 B 需更低权重（λ0.05）/去掉。

## 3. 临时止血（可选，不等重训）

泛化验证/论文先用 **v7（MoMPNN）与 v9（LigandMPNN）**作主方法（已可用）；v10/v11 按上面矩阵
推进，成功后替换。v10 的负向超线性可写成论文"失败/机制"章节（配响应曲线图：target→生成电荷）。

## 4. 文件

- 诊断脚本：`v10_diag_response_curve.py`（服务器 index/v10_repair/ 已有、已跑 17 蛋白）
- 训练补丁：`train_finetune_v11_patch.md`（A/B/C 三处修改 + 命令）
- 本 README：执行总览（诊断版）

---

## 5. v11 消融结果（2026-08-29 闭环，四版全未达标）

| 版本 | 配置 | valid 区内 slope | 结论 |
|------|------|:---:|------|
| v10 | A相对±12 + B + C横批 | 1.67 ± 0.53 | 退化基线 |
| **v11a** | A相对±12 + C，去B | **1.41** ± 0.47 | 四版最好 → **B 纯有害** |
| v11b | A绝对[−35,20] 单开 | 1.50 ± 0.42 | A-fix 无效，且拖坏好蛋白（1C6O 0.98→1.39）|
| v11c | A绝对 + B-fix + C-fix | 1.57 ± 0.37 | 全fix 未达标 |

**判据**：区内 slope ∈ [0.9,1.15] → **四版全超，无一通过**。
**归因**：① B(L_add) 纯有害（去 B 改善 0.26，加回反而差）；② A-fix 只是"平均化" slope 非校正；③ C-fix 无帮助；④ 残余过冲主因 = decouple 机制本身/编码器响应增益。详见 `analysis/report/2026-08-29_v11b_c_compare.md`。

## 6. bias 排查结论（2026-08-29 代码确认）

- `conditioned_sampler.conditioned_sample` **不传 bias_callback**，且 `fd["bias"]=torch.zeros` → **Phase 3 主路线无 logit bias**（条件是 soft prompt 注入 h_V）。
- 诊断脚本（`v10_diag_response_curve.py` L159）用 `conditioned_sample` → **slope>1 不在 bias 公式**，而在 **ConditionEncoder 学到的响应增益**（target→电荷 映射斜率 >1）+ 解码放大。
- **推论**：推理侧可做**条件向量标量校准**（喂编码器前把 target 校正），零重训成本。

## 7. v12 方案（推荐，吸收 v10/v11 全部教训）

### 7.1 推理侧条件向量校准（零重训，先做，最可能免费解决 slope>1）

用诊断 slope(a)/截距(b) 校准喂给编码器的 target：
```
target' = (target − b) / a
```
- 复用 `output/v1{0,1a,1b,1c}_diag_response.json` 已拟合的 (a,b)；per-protein 或全局皆可
- 实现：`run_guided.py` 加 `--calibrate {off|global|per-protein}`，读校准表
- 验证：校准后跑同一诊断 → slope 应回落到 ≈1
- 本质 = v9 时代 `charge_calibration` 的复活，但这次有完整诊断数据

### 7.2 训练侧 v12（若校准不够或要根治"删减捷径"）

```
L = CE
  + λ_c   · | Σ_surface q_i − (target − Q_core_native) |      # 表面电荷密度目标（核心锁死 native）
  + λ_comp· relu(native D/E_sur − gen D/E_sur) + relu(native K/R_sur − gen K/R_sur)  # 组成双计数
  + λ_grav· relu( surface_GRAVY(gen) − surface_GRAVY(native) − margin )             # 堵删减
  + λ_anchor· | Q(gen|target=native) − Q_native |              # native 锚（防响应漂移）
  + λ_keep· seq_keep                                           # 原有
```
- **表面掩码**：骨架固定 → rASA≥0.25 掩码预计算为**常数**（fractional SASA 工具已有，P0）
- 可选 **L_gain** 增益监督 `|(Q(t1)−Q(t2))/(t1−t2) − 1|`（双前向，成本×1.5~2，先不做）
- **教训对应**：组成双计数治"只盯 D/E 漏 K/R"（L_add 缺陷）；GRAVY 治"无差别删带电残基总数"（v7/v9 根因）；anchor 治"decouple 弱化 native 锚"；核心锁死治"无差别删核心"。

### 7.3 验证闭环（判据更新）

- 闭环诊断：区内 slope ∈ [0.9,1.15]（不变）
- **新判据（治删减捷径）**：生成序列 D/E、K/R 计数 **≥ native**（证明"加"不是"删"）；表面 GRAVY ≤ native + margin
- 完整泛化验证：折叠 TM-score 兜底（加电荷后不伤折叠）
- 物理自洽：PROPKA（H4，已有）

### 7.4 执行顺序

1. **先 7.1 校准**（零成本，立即可验）→ 若 slope 回落，先交付校准版跑泛化验证
2. 再 7.2 训练 v12（30ep）→ 闭环诊断 + 新判据（D/E/K/R 计数、GRAVY）
3. 达标 → 完整泛化验证；不达标 → 消融 λ_grav / λ_comp / margin

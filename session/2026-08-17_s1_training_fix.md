# 第十四轮（2026-08-17）— 训练修正：治 S1 注入选择性

> 状态：训练已启动，结果待补。本文件记录设计、代码修改、冒烟与训练参数（及时复盘用）。

---

## 一、背景：为什么修 S1

第十三轮（n=20）按判断标准 v1 重新评估，**S1 注入选择性明确失败**：
- A 场景（target=原生电荷、无改 pI 需求）下，条件臂 vs 基线 identity 仅 **0.45–0.59**
  （要求 ≥0.7）——模型"无需求时也大幅重写"（>50% 位点非保守替换）。

**机制**：训练混合目标里 50% 样本被扰动（target=native±Uniform[1,4]），模型学会
"见电荷条件就重写"的全局规律，泛化到自洽样本。且现有 KL 锚只约束**分布距离**
（softmax 概率 0.30→0.29 时 KL 已很小，但 argmax 可能翻盘 K→R），管不住序列级
identity 下降。

## 二、修正设计（两板斧）

### 修正 A：扰动比例 50% → 30%（原生标签 70%）
- `--perturb_prob` 默认 `0.5 → 0.3`
- 让"target=原生时保持"成为主导训练信号；30% 扰动样本保留电荷偏移学习能力
  （避免模型完全丧失偏移能力）

### 修正 B：序列保持正则（seq-keep）——S1 判据的训练侧直接对应
- 以**无条件 argmax 序列**为锚（`argmax(logits_uncond)`，冻结 backbone 预计算，常数）
- 对**自洽样本**（未扰动）施加 `CE(logits_cond, anchor)`——条件输出逐位逼近无条件输出
- **只在自洽样本施加**；扰动样本 target≠native 时电荷偏移是期望行为，不受约束
- 为什么比 KL 更直接：KL 管分布距离，管不住 argmax 翻盘；seq-keep 直接惩罚
  "无条件最可能氨基酸在条件分布下失势"，对应序列级判据

**新损失**：`L = CE + λ_c·charge_deviation + λ_kl·KL + λ_keep·SeqKeep`
（λ_keep 默认 0.5，新参数 `--lambda_keep`）

## 三、代码修改清单

| 文件 | 修改 |
|------|------|
| `code/src/losses.py` | 新增 `sequence_keep_loss()`（含完整 docstring：与 KL/CE 的语义区别、施加时机） |
| `code/train_finetune.py` | ① docstring 损失/混合目标更新；② `--perturb_prob` 默认 0.3、新增 `--lambda_keep`(0.5)；③ import `sequence_keep_loss`；④ 预解析加 `dom["seq_anchor"]`（无条件 argmax，X 位置锚到 0）；⑤ 训练循环 mask_p 初始化外移 + keep 计算（仅自洽样本）+ total 并入；⑥ epoch 汇总/log/checkpoint 加 keep 项与追溯字段 |

## 四、Bug 修复（冒烟发现）

**`gather` 维度错误**：`featurize` 返回的 `S` 带 batch 维 `[1,L]`，
`torch.where(dom["S"] < 20, ...)` 把 `[L]` 锚广播成 `[1,L]` → `seq_anchor` 变 `[1,L]`，
再 `unsqueeze(0)` → `[1,1,L]` → gather 维度不匹配。
修复：`dom["S"][0]` 取单链索引（见注释）。

## 五、冒烟结果（3 域 1 epoch）

```
epoch 1/1  total=5.0221  ce=1.6543  charge=6.2905  kl=0.0036  keep=0.4448
```
keep 有值（0.4448）→ 序列保持正则正常参与训练。

## 六、训练启动

```
PID 1800662（nohup setsid）
python code/train_finetune.py --device cuda:1 --epochs 30 \
  --perturb_prob 0.3 --lambda_keep 0.5 --charge_temp 0.5 \
  --out_dir code/output/finetune_s1
```
- 数据：999 结构域 × 8 pH = 7992 样本
- 日志：`code/log/train_s1.log`；进度：`code/log/train_progress_s1.json`
- 上次同类训练 30 epoch ≈ 14.8min

## 七、训练结果（30 epoch 完成，16.5min）

```
epoch 30/30  total=3.2690  ce=1.8564  charge=1.9690  kl=0.1342  keep=0.8428
```
- **ce 1.856**（结构锚稳定，≈上轮 1.86）
- **charge 1.969**（上轮 finetune_t05 为 1.58——seq-keep 部分占用容量，电荷损失略高，但复验 H2 仍 6/8 达标）
- **keep 0.843**（全程稳定——序列保持正则主导，S1 修正生效）

产物：`code/output/finetune_s1/condition_encoder_last.pt`（30 epoch checkpoint 全存）

## 八、n=20 复验（新编码器）—— 即时检查

### S1 注入选择性（cond vs base identity，从 fasta 直接算）

| PDB | A 场景 | 上轮 t05 | B 场景 |
|-----|:---:|:---:|:---:|
| 1BC8 | **0.516** ± 0.063 | 0.45–0.59 | 0.596 ± 0.041 |
| 1CRN | **0.668** ± 0.063 | — | 0.698 ± 0.056 |
| 1UBQ | **0.582** ± 0.045 | — | 0.605 ± 0.051 |
| 2LZM | **0.565** ± 0.035 | — | 0.647 ± 0.036 |

**结论**：修正方向正确（0.45–0.59 → 0.52–0.67，1CRN 逼近 0.7），但 **A 场景未达 ≥0.7 阈值**。
且 B 场景 identity ≥ A 场景——模型仍"无差别重写"（电荷偏移与无需求改写未充分区分），只整体略降。
S1 是**软判据**（判断标准 v1，不单独判 FAIL），但作为本轮修正目标未完全达标。

### H2 电荷命中（|实际−target|≤2.0）

| PDB | A_cond | target | Δ | B_cond | target | Δ |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|
| 1BC8 | +9.47 | +9 | 0.47 ✅ | +13.19 | +14 | 0.81 ✅ |
| 1CRN | −1.08 | −1 | 0.08 ✅ | +2.45 | +4 | 1.55 ✅ |
| 1UBQ | −0.55 | 0 | 0.55 ✅ | +3.79 | +5 | 1.21 ✅ |
| 2LZM | +11.09 | +8 | 3.09 ❌ | +19.45 | +13 | 6.45 ❌ |

**6/8 达标**。seq-keep 未削弱电荷偏移能力；2LZM 过冲为老问题（上轮 +13.24 → 本轮 +11.09，
略改善未根治）。

### 四指标打分
- pLDDT/TM/%sol：16/16 臂全部完成 ✅
- TemBERTure：SSL 错误（huggingface 联网校验证书失败）→ **离线模式 `HF_HUB_OFFLINE=1` 重跑**（模型在本地缓存），已完成 1 臂验证，并行重跑中

## 九、待续（打分 + TemBERTure 完成后）

1. 跑 `phase3_antidrift_n20_stats.py`（配对检验 + 判定）
2. 按判断标准 v1 最终判定（H1 TM-score / H2 / H3 / S1）
3. 完整报告 + PROJECT_STATUS + DOCUMENT_INDEX + git push

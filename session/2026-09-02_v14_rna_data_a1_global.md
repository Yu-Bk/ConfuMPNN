# 会话记录 — v14 配体重启：RNA/DNA 结合蛋白数据扩充 + A1 全局化 + 重训（2026-09-02）

> **状态**：训练运行中（GPU4，50ep，进程 1565568）。
> **数据**：`data/ligand_train/labels_v14_merged.npz`（5148 域 = 旧 4957 + RNA/DNA 191）
> **报告**：`analysis/report/2026-09-02_v14_rna_data_a1_global.md`
> **设计**：`index/PROJECT_LOCAL_V12_2.md`（§7 + 本会话）

## 一、背景与决策链

用户重启配体线（决策 D 曾停止）。方向 = 补 RNA/DNA 结合蛋白数据 + 让 A1 起作用/全局化 + 重训。
主 session 追加两条硬指令：
1. **number_of_ligand_atoms 16→25 全部修正**（权重 `ligandmpnn_v_32_010_25.pt` 的 atom_context_num=25，
   原版 LigandMPNN 预训练即 25；项目代码写 16 是与权重不匹配 bug，可能是配体 context 不足的隐藏因素）。
2. **数据源优先核糖体结合蛋白**（4V4T/9RVC 等 70S 拆链，全部并入 + 序列去重）。

epoch 决策（用户）：配体线此前 30ep 未收敛（v13 末 3 ep charge 仍缓降）→ 本轮变化更大
（RNA 新类型 + 25 原子首次）→ **epochs=50**，监控 plateau。

## 二、数据收集与拆链

### 2.1 来源
- 核糖体（主源）：**4V4T**（T. thermophilus 70S, 49 蛋白链）、**9RVC**（47 链）、**4YBB**（99 链，约 2 拷贝）
- 补充（RNA/DNA 结合复合物，多样）：5AVC(核小体)/3WVK(HindIII)/1BP7(I-CreI)/8ZDR(Cas9d)/5VVL(Cas1-Cas2)/
  2V3C(SRP)/2ZZN(aTrm5-tRNA)/3ADB(tRNA 激酶)/5GIN(box C/D RNP)/7V9X(retron)/9FB4(SV40)/3HOT(Mos1)/
  4NOD(TFAM)/7OUH(intasome)/6IFL/9ASH(Csm)
- 随机池补充池：RCSB 蛋白+核酸复合物搜索（resolution≤3.5, 聚合实体≤12）→ 800 采样 → 779 下载 →
  prody 筛选出 570 可用 PDB / 1509 可用单链候选（`_screen.json`）

### 2.2 拆链逻辑（`code/tests/ligand_v9/split_nucleic_complex.py`）
- 每个蛋白链单独成 PDB：蛋白链 id 重标 A，配体原子重标 Z
- 配体保留 = 该链任一重原子 **15Å 内**的非蛋白非水原子（RNA/DNA/小分子/配位离子）
- QC：写完 `parse_PDB` 验证 L 匹配 + N/CA/C/O mask ≥0.9 + Y 原子数达标
- 拆出 **260 个**单链样本（核糖体 185 + 补充 75）

### 2.3 标签构建 + 去重（`build_rna_v14_labels.py`）
- 8 pH（uniform 4-10）+ net_charge + pI，同 build_ligand_labels
- **序列精确去重**（跨结构核糖体同源蛋白、核小体组蛋白多拷贝）→ 191 唯一域
- 排除与旧训练集序列重复
- 产物 `labels_rna_v14.npz`（191×8）→ 合并旧集 → **`labels_v14_merged.npz`（5148 域 × 8 = 41184 样本）**
- all_pdb 加 191 个新 symlink（现 5163）

### 2.4 新 RNA 域组成
- 来源：4V4T 46 / 4YBB 58 / 9RVC 44（核糖体 148）+ 一般 43
- L 50-446，median 119；300-500 有 12 个（长蛋白覆盖）
- charge@7.4 mean +8.7（碱性，符合核酸结合蛋白），min -14.9 / max +40.7
- K/R mean 25.3 / D/E mean 16.7

## 三、number_of_ligand_atoms 16→25（全配体脚本修正）

改的文件：train_finetune.py（默认+featurize）、run_guided.py（默认+featurize）、
validate_generalization.py / transfer_validation.py（--num_ligand_atoms 默认 25）、
smoke_guided.py、sample_unconditioned_ligand.py、raw_ligandmpnn_pocket.py、
ligand_pocket_validation.py、mompnn_compat_test.py。
protein 模式 number_of_ligand_atoms=0 不动。权重实测 atom_context_num=25，smoke_guided 通过。

## 四、A1 全局化（`--pocket_mode global`）

### 4.1 为什么 v13 无效（已证）
v13 A1 只护 pocket（Cα-配体<8Å），非 pocket surface 仍删（frac_floor 0.5 允许删一半）→
组成 8/10 删 0.55-0.69×、Tm/Sol 恶化 17/50。根因 = frac_sasa 盲区（深口袋 frac<0.25 不算表面）。

### 4.2 改动
- `v12_losses.py::pocket_count_loss`：支持任意 mask + `normalize`（除以 native 计数 → 分数化，
  global 区域大不随蛋白长度膨胀）+ N=0 死锁保护（min_abs_cap=2）
- `train_finetune.py`：
  - `--pocket_mode {keep,free,global}`（原 keep/free 保留）
  - `global` 模式：计数区 = **charge_surf_mask = surface ∪ pocket**（三块互斥分区的"温和改"残基）
    → 绕开 frac_sasa 盲区，直接锚"带电残基总数"命脉
  - 三块互斥分区（core/pocket/surface）在 keep/global 都算；A2 `surface_charge_target_loss`
    extra_mask=charge_surf_mask 在 global 也生效（core 锁 native，无双算）
  - 每方向双向计数 `relu(N·floor−gen)+relu(gen−N·ceil)`

### 4.3 超参选择（记录决策）
- **floor 0.8**（native 计数区的 80%；配体删减实测 0.53-0.69×，0.8 触发防护留 20% 温和变化余量）
- **ceil 1.3**（防成对加/无限加，v12 只设下限→过度添加 1.5-2× 的教训）
- **λ_pocket 0.3**（比 v13 0.2 高，但 global 已 normalize 到 O(1) 量级，避免压崩 CE/charge；
  若监控显示 charge 命中差再降）
- **cutoff 8.0**（与 define_pocket.py/验证口径一致）
- normalize=True 只用于 global（keep 模式保持 v13 原始量级语义，可复现）
- 已知张力：对强碱性 RNA 蛋白做"极端反号 target"（如 native +25 → −20）会被 per-sign ceil
  限制——但这本身是合理物理约束（强碱性核糖体蛋白不应翻成强负电）；验证 arms 为 native±8 不受影响。

### 4.4 dry-run 结果
- 50 域混合（20 旧 + 30 RNA）：0 NaN、0 分区失败、checkpoint 生成
- RNA 蛋白（4V4T_AB 等）三块分区正常（如 4V4T_AB pocket=37/234 core=100 surface=116）

## 五、训练启动（进行中）

命令摘要（log/v14_ligand_train.log）：
```
--ligand --weights ligandmpnn_v_32_010_25.pt
--labels data/ligand_train/labels_v14_merged.npz --dompdb data/ligand_train/all_pdb
--out_dir output/finetune_ligand_v14_rna --device cuda:4 --epochs 50
--decouple_absolute --decouple_abs_lo=-35 --decouple_abs_hi=20
--v12_supervision --frac_floor 0.5 --gravy_margin 0.4 --lambda_v12 0.2 --lambda_target 0.2
--sasa_threshold 0.25 --ph_aware_filter --structure_boost 1.5
--charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15
--lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5
--pocket_mode global --pocket_cutoff 8.0 --pocket_floor 0.8 --pocket_ceil 1.3 --lambda_pocket 0.3
```
- 进程 1565568（GPU4 cuda:4），启动确认存活 + GPU 占用正常 + [A1 global] 日志确认
- 数据：5148 域 × 8 pH = 41184 样本
- 预计 50ep × ~15-19min ≈ 14-16h（预解析 SASA 5148 域 ~1h 先行）

## 六、验证集重构（v14 配体模式）

`data/validation_pdbs/validation_manifest_v14_ligand.json`（13 蛋白）：
- **保留单体**小分子/核苷酸配体蛋白：1AZM/1AS2/2FEO/5CQH/1CGE/1A65/1BJ4
- **删除同源二聚体**：1C6O/1AXW/1AG0
- **新增 held-out 核酸结合**（序列不在 labels_v14_merged，防泄漏）：
  21KL_A(RNA/DNA)/3MXB_A(DNA meganuclease)/4GDF_A(DNA,497)/5ZR1_B(DNA,374)/
  8DR1_A(DNA junction,493)/9DWG_L(DNA Polβ,323)
- 验证链待训练完成后跑：组成 → slope → 泛化(H2/H1/H4) → H3 → Tm/Sol

## 七、训练启动确认（2026-09-02 16:33 更新）
- 预解析完成：**5147 域**（1 坏域跳过，prody 无法解析），缓存 encode ~23.69GB
- **epoch 1/50 完成：0 NaN**，total=6.27 ce=1.42 charge=6.02 kl=0.09 keep=0.66
  （v13 末 3ep charge 3.6-3.67 缓降 → 本轮 50ep 目标收敛）
- 节奏：~16 min/epoch → 50ep ≈ 13-14h；GPU4 显存 ~30GB
- RNA 域（4V4T/9RVC/4YBB 等）预解析分区全部正常（如 9RVC_x pocket=26/65 core=9 surface=42）
- 全程警告 4 条（旧小分子域含 UNK 触发 freesasa 失败 → 跳过 v12 监督，v13 同行为，不致命）

## 八、验证集修订 + 一致性分析 + 数据充分性（2026-09-02 下午，用户新策略）

> v14 训练（epoch1/2）被主 session 暂停作废。按用户策略重构数据 + 验证集后重启。

### 8.1 数据充分性评估（Q3，实测）
候选池（RCSB 蛋白+核酸复合物，779 下载/570 可用复合物）1468 条可用单链：
- **DNA 1057 / RNA 364 / hybrid 47**
- 去重 + 排除当前训练序列后：**可用未见链 939** = RNA 232 / DNA 676 / hybrid 31
  L 覆盖 80-500，>300：RNA 93 / DNA 173 / hybrid 13；median ~201
- 结论：**数量充足，无需外部数据库**。训练补充直接从内部池取（免下载）。

### 8.2 训练集变更（Q3 执行）
在旧 4957 + RNA/DNA 209 基础上（见 §8.5 规模），新补充 = 209 中净增 18（相对 §2.3 的 191）：
- 转训练 DNA(3)：**4GDF_A(L497)/8DR1_A(L493)/5ZR1_B(L374)**——3 个最长、大复合物 DNA 结合
- DNA 补充（同复合物其他亚基，免新拆）：8DR1_B/C/D/E/F/G/H（RFC/PCNA 亚基）+ 5ZR1_A/D/E/F（ORC 亚基）→ 共 14 条不同 DNA 结合蛋白
- RNA 补充（非核糖体，新拆）：1G59_A（glutamyl-tRNA synthetase，L468）、5W5H_A/C（NSun6/tRNA，L451/446）、8VIV_A（FBF-2 Pumilio，L401）
- 拆链脚本同 `split_nucleic_complex.py`，QC 全过

### 8.3 验证集重构（Q1+Q3，最终 11 蛋白）
`data/validation_pdbs/validation_manifest_v14_final.json`
- **删**：1C6O/1AXW/1AG0（同源二聚体）；1AZM（泄漏：序列=训练 `1HCB.pdb` 完全一致）
- **替换 1AZM → 6D2O**：beta 碳酸酐酶 + 4-methylimidazole（有机小分子），L209，表外（RCSB 现下），
  序列核对不在 labels_v14_final（Q1 确认；人类 CAII 成熟序列 1AM6/1BNN 等全泄漏，故选异源 beta-CA）
- **核酸 4**：RNA = 21KL_A(hybrid,L237) + 2E9R_X(FMDV RdRp 纯 RNA,L476)；DNA = 3MXB_A(meganuclease,L153) + 9DWG_L(Polβ,L323)
- **非核酸 7**：6D2O/1AS2/2FEO/5CQH/1CGE/1A65/1BJ4
- **11 个验证蛋白序列全部 leak=False**（对 labels_v14_final 重跑核对，§8.6 表格）

### 8.4 训练/验证种类与长度一致性（Q2）
训练集 labels_v14_final（5166 域 = 旧 4957 + RNA/DNA 209）：
- 类别（按 all_pdb symlink 源）：small_mol 4145 / metal 564 / 旧 rna(核苷酸辅因子) 242 / 旧 dna 6 / rna_pdbs(真核酸链) 209
- 长度：旧 L 20-500 median 297；RNA/DNA 209 域 L 50-497 median 122，>300 有 26
- RNA/DNA 真核酸链域占比 **4.0%**
验证集 11 蛋白：small_mol 1 / nucleotide 3 / metal 1 / long 2 / RNA 2 / DNA 2；L 153-504 median 237；核酸占 **36%**

**判断**：验证集对 RNA/DNA 的 36% 远高于训练集 4%，是**有意的"新能力过采样"**——合理：
1) 目的就是检验新增 RNA/DNA 结合能力是否真正学会，若按训练比例（~0.5 个核酸蛋白≈0）则完全测不到；
2) 但 4.0% 训练占比意味着 RNA/DNA 能力仍是小样本（尤其 DNA 域 ~25，远小于 ribosome RNA ~150），
   验证可能暴露 DNA 欠拟合——这正是本轮要测的，失败即为"数据不足"信号（可用 §8.1 的 939 未见链补训）。
3) 所以这个验证集应解读为"能力 check"，不是总体命中率估计；RNA/DNA 判据单独看。

### 8.5 重建标签（label 段错位 bug 教训）
- 不 append：在 `rna_pdbs/`（279 split 文件，含原 260 + 新 19）上**整体重跑** `build_rna_v14_labels.py`
  → `labels_rna_v14_sup.npz`（209 唯一域 ×8 = 1672 样本，去重+排旧集自动完成）
- 合并旧 `labels.npz`(4957) → **`labels_v14_final.npz`（5166 域 ×8 = 41328 样本）**
- sanity：pH/charge 长度 = 8×域数 ✓；collision=0 ✓；all_pdb +18 symlink（现 5181）

### 8.6 最终验证蛋白泄漏核对（对 labels_v14_final）
| 蛋白 | L | leak | | 蛋白 | L | leak |
|---|---|---|---|---|---|---|
| 6D2O | 209 | False | | 1A65 | 504 | False |
| 1AS2 | 312 | False | | 1BJ4 | 470 | False |
| 2FEO | 221 | False | | 21KL_A | 237 | False |
| 5CQH | 183 | False | | 2E9R_X | 476 | False |
| 1CGE | 162 | False | | 3MXB_A | 153 | False |
| | | | | 9DWG_L | 323 | False |

### 8.7 dry-run + 重启训练
- dry-run（`labels_smoke_v14_final.npz` 50 域=25旧+25RNA/DNA，global）：**0 NaN、0 分区失败**，checkpoint 生成
- **训练重启**：GPU4 cuda:4，`output/finetune_ligand_v14_rna/`（原 epoch1/2 已清），50 epochs，
  超参同前（pocket_mode global floor0.8/ceil1.3/λ0.3 + v12 全套），`labels_v14_final.npz`（5166域），
  日志 `log/v14_ligand_train.log`（append，含两次启动标记）

## 九、核酸数据大规模扩充 + v14 二次重启（2026-09-02 用户决策）

> v14 训练（第 2 次，epoch≈0）被暂停。用户：既然候选池储备充足（939 未见链），应趁损失≈0
> 补齐 RNA/DNA 短板（DNA 35→≥100；非核糖体 RNA 补足）再训，避免 13h 后欠拟合再重训。

### 9.1 扩充选择（实测，来自 923 未见唯一链）
自动化精选（按类型/分辨率≤3.5/家族多样性，排除核小体组蛋白冗余、核糖体、验证同源家族 meganuclease&Polβ/X-family 标题）
- **DNA 120 候选链**（155 复合物→拆出后按序列去重），功能覆盖：EcoRV/BamHI/PvuII/McrBC 限制酶、
  Cre 重组酶、IS608/ISDra2/Mos1 转座酶、MutH/AlkB/TDG/MutY/AGOG/AlkD 等 DNA 修复酶、
  聚合酶 eta/iota/Rev1(Y 家族)、Pif1/NS3 解旋酶、M.TaqI/DRM2/CpG 甲基转移酶、p53/CTCF/Smad4
  转录因子、UP1/端粒、Argonaute/Piwi、FEN1/Artemis/ExoI 核酸酶 等
- **RNA 85 候选链**（非核糖体），功能覆盖：RNase III/RNase E/RNase T/NSP15(CoV2) 核糖核酸酶、
  tRNA 合成酶(1B23)/tRNA 修饰酶(NSun6/SepSecS/RlmJ/PUS1)/CCA-adding、Dengue/ZIKV NS3 解旋酶、
  MS2/TMV/SM 衣壳蛋白、Pumilio/FBF-2/Dicer/ADAR、CRISPR(Cascade/Cas12k/Csm)、HIV-RT 等
- 全部经 split_nucleic_complex.py 拆链（单链 A + 15Å 核酸配体 Z）+ QC（150/150 复合物，0 真失败）
- 序列精确去重 + 与当前训练集精确去重 → **RNA/DNA 唯一域 209 → 414**（净增 205）

### 9.2 最终 RNA/DNA 414 构成（实测）
- **DNA 155**（含 hybrid 3；L med 255，>300 = 61）
- **RNA 非核糖体 108**（L med 276，>300 = 50）
- RNA 核糖体 148（4V4T/9RVC/4YBB）
- 来源 Top：4YBB 58 / 4V4T 46 / 9RVC 44 / 8DR1 8 / 5AVC 7 / 5VVL 6 / 6IFL 6 / 5ZR1 5 / 9ASH 5 ...

### 9.3 标签重建（不 append，整体重跑）
- rna_pdbs 现 484 个拆链文件 → 整体重跑 build_rna_v14_labels.py → `labels_rna_v14_sup2.npz`（414 域 ×8）
- 合并旧 4957 → **`labels_v14_final.npz`（5371 域 ×8 = 42968 样本；RNA/DNA 占比 7.7%）**
- sanity：pH/charge 长度 = 8×域数 ✓；collision=0 ✓；all_pdb 补 205 symlink（现 5386）
- 全部 414 新域 parse 预检通过（build 即 parse QC）

### 9.4 验证集 11 蛋白泄漏核对（对 labels_v14_final 5371）
| 蛋白 | 新训练含验证序列? | | 蛋白 | 新训练含验证序列? |
|---|---|---|---|---|
| 6D2O | False | | 1A65 | False |
| 1AS2 | False | | 1BJ4 | False |
| 2FEO | False | | 21KL_A | False |
| 5CQH | False | | 2E9R_X | False |
| 1CGE | False | | 3MXB_A | False |
| | | | 9DWG_L | False |
→ **11 个验证蛋白全部 held-out（无任何新训练域与之同序列）**

### 9.5 dry-run + 训练重启
- dry-run（`labels_smoke_v14_final2.npz` 25 旧+25 新核酸，global）：**0 NaN、0 分区失败**，checkpoint 生成
- **训练重启（第 3 次）**：GPU4 cuda:4，PID **1959542**，`output/finetune_ligand_v14_rna/`，
  50 epochs，`labels_v14_final.npz`（5371 域），日志 `log/v14_ligand_train.log`（append）
  超参同前（global floor0.8/ceil1.3/λ0.3 + v12 全套 + 25 配体原子）。预解析 5371 域预计 ~45min，首 epoch ~16-18min

### 9.6 首 epoch 确认（2026-09-02 19:21）
- 预解析完成：**5370 域**（1 坏域跳过，prody 无法解析），缓存 ~24.7GB
- **epoch 1/50：0 NaN**，total=6.22 ce=1.45 charge=5.88 kl=0.11 keep=0.70；~16.9 min/ep → 50ep ≈ 14h
- GPU4 显存 ~25GB；进程 PID 1959542

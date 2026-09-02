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

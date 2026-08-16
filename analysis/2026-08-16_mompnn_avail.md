# MoMPNN 可用性调研（Stage E0）— 2026-08-16

> 对应 `index/PROJECT_EXTEND.md` Stage E0。目标：确认第二版路线 A（复用开源 MoMPNN 权重）是否可行。

## 结论（TL;DR）

**路线 A 可用（需小幅适配接入方式）** ✅

- MoMPNN 权重 = **纯 backbone ProteinMPNN**（不含配体上下文），与 LigandMPNN 源码中的 `ProteinMPNN` 类在 `model_type='protein_mpnn'` 模式下**逐层完全匹配**，`load_state_dict(strict=True)` **直接通过**。
- 已用 1BC8.pdb **实际生成序列**，seq_rec ≈ 0.45（与 LigandMPNN 基线 0.49 相当），模型工作正常。
- 唯一需要改动的是把 `run_guided.py` 中硬编码的 `model_type='ligand_mpnn'` 参数化（适配工作量小）。

---

## 1. 仓库情况

- 2026-08-16 clone 自 `github.com/Qivon7/MoMPNN`，存放于 `ConfuMPNN/MoMPNN/`（项目根，已 gitignore）
- 论文：**Property-Driven Protein Inverse Folding with Multi-Objective Preference Alignment**（ICLR 2026）
- 仓库内容：`README.md` + `mompnn_paper_checkpoints/`（8 个 `.ckpt`，共 ~53 MB）
- ⚠️ **该仓库只有权重，没有训练/推理代码**。README 声明权重与 ProteinMPNN 官方格式完全兼容，推荐直接用 LigandMPNN 的推理管线加载。

## 2. 权重格式

| 项 | 结论 |
|----|------|
| 保存格式 | PyTorch 权重（zip 容器），非 LFS 指针 |
| 内部 key | `model_state_dict` / `num_edges=48` / `noise_level=0.2` —— **与 LigandMPNN 官方权重格式完全一致** |
| dtype / 规模 | 全部 fp32，118 个张量（≈165 万参数，标准 ProteinMPNN 规模） |
| 命名解读（推测） | `mompnn_`=单目标 / `modpo_`=多目标 DPO；`protsol`=可溶(Protein-Sol)、`temberture`=热稳(TemBERTure)；`ig`=逆折叠、`tm`=熔解温度；`esm`=含 ESM 特征；`b01`=DPO β=0.1 |

8 个权重（全测过）：
`mompnn_protsol_ig`、`mompnn_temberture_tm_esm`、`modpo_protsol_ig_esm`、`modpo_protsol_tm`、`modpo_protsol_tm_esm`、`modpo_temberture_ig`、`modpo_temberture_ig_esm`、`modpo_temberture_tm`

## 3. 兼容性测试（confumpnn 环境，torch 2.2.1）

对全部 8 个权重逐一测试两种 `model_type`：

| 测试 | 结果（8/8 一致） |
|------|------------------|
| **`protein_mpnn`（纯 backbone）strict=True** | ✅ **全部 PASS**（无 missing / unexpected） |
| `ligand_mpnn`（配体模式，我们管线现状）strict=False | missing=79, unexpected=0 |

- `ligand_mpnn` 模式下 missing 的 79 个层全是**配体上下文层**：`features.node_project_down/norm_nodes/type_linear/y_nodes/y_edges`、`W_c/W_nodes_y/W_edges_y/V_C/V_C_norm/context_encoder_layers` 等。
- `unexpected=0` 说明权重里**没有多余层**，纯粹是"模型有、权重缺"的方向，即纯 backbone。

**结论**：MoMPNN 权重**不含配体上下文**，是标准 ProteinMPNN（模型类名 `ProteinMPNN`，`model_type='protein_mpnn'`）。

## 4. 前向验证（实际生成序列）

- 输入：`code/input/1BC8.pdb`（93 残基）
- 加载：`model_type='protein_mpnn'`, `atom_context_num=0`, `k_neighbors=48`, `load_state_dict(strict=True)`
- featurize：`use_atom_context=False, number_of_ligand_atoms=0, model_type='protein_mpnn'`
- 结果：温度 0.3 采样 2 条，**seq_rec ≈ 0.452 / 0.462**（LigandMPNN 基线 1BC8 为 0.4946，处于正常范围）

## 5. 接入现有管线的适配点

`code/run_guided.py` 现状 vs MoMPNN 需求：

| 位置 | 现状 | 需要改为 |
|------|------|---------|
| `load_model()` | 硬编码 `model_type='ligand_mpnn'`，从 ckpt 读 `atom_context_num` | `model_type` 参数化；纯 backbone 时 `atom_context_num=0` |
| `main()` featurize | `use_atom_context=True, number_of_ligand_atoms=16` | `use_atom_context=False, number_of_ligand_atoms=0` |

- 工作量：**小**（改 `load_model` + featurize 分支，加一个 `--model_type` / `--backbone` 参数）
- 注：MoMPNN checkpoint 用 `model_state_dict` / `num_edges`，与 LigandMPNN 官方 .pt 完全同构，无需特殊处理（甚至比预期的 PL `state_dict` 更省事）

## 6. 待办（下一步 E1）

1. `run_guided.py` 加 `--weights` 已支持；补 `--model_type protein_mpnn` + `--use_atom_context False`，使 MoMPNN 权重可直接跑
2. 对照实验：同一 PDB/pH 下，MoMPNN（多目标 DPO）vs 原版 LigandMPNN 的 pH 响应与可设计性（ESMFold pLDDT）
3. 三个目标（可设计/热稳/可溶）打分：ESMFold（`confumpnn-esmfold` 环境）+ Protein-Sol + TemBERTure

## 附：测试脚本（已归档）

- 兼容性：`code/tests/mompnn_compat_test.py`（8 权重 × 2 模式 load_state_dict）
- 前向：`code/tests/mompnn_forward_test.py`（MoMPNN 权重 + 1BC8.pdb 采样）

运行（confumpnn 环境）：`conda run -n confumpnn python code/tests/mompnn_compat_test.py`

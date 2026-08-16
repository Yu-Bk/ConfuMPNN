# ConfuMPNN 配置文档

> 全部配置项的**含义、默认值、建议范围**。配置分布在 `code/configs/*.yaml` 与 `run_guided.py` 的命令行参数。

---

## 一、结构过滤器预设（`code/configs/filter_presets.yaml`）

### 1.1 通用字段说明

每条规则是一个字典，字段含义：

| 字段 | 含义 | 单位 |
|------|------|------|
| `radius` | 空间邻域半径（欧氏距离） | Å |
| `threshold` | 触发抑制的数量阈值 | 个数 |
| `strength` | 抑制强度（加到 logits 上的负 bias；越负抑制越强） | 无量纲 |
| `burial_radius` | 判定"核心埋藏"的邻域半径 | Å |
| `charge_radius` | 核心规则里统计带电残基的半径 | Å |
| `burial_threshold` | 判定"埋在核心"的 burial 比例（10Å 内 Cα 数 / 最大值），>该值视为核心 | 比例 0–1 |
| `charge_count` | 核心内带电残基数阈值 | 个数 |

### 1.2 四条规则与阈值来源

阈值来自 CATH S40 1000 结构域统计的 **99 分位**（详见 `docs/TECH.md` §3.3 与 `analysis/report/2026-08-16_phase1_examples.md`）：

| 规则键 | 检测内容 | 关键阈值 |
|--------|---------|---------|
| `charge_cluster` | 10Å 邻域内同号强电荷（K/R 或 D/E） | `threshold: 6` |
| `salt_bridge` | 10Å 内正负电荷对（min(正,负)） | `threshold: 4` |
| `core_charge` | 核心埋藏位置 8Å 内带电残基 | `charge_count: 6` |
| `same_sign_cluster` | 8Å 邻域同号电荷 | `threshold: 4` |

### 1.3 四个预设（`--preset` 选择）

| 预设 | 设计意图 | 与 default 的差异 |
|------|---------|------------------|
| `default` | 通用默认（99 分位阈值） | 基准 |
| `nucleic_acid_binding` | 核酸结合蛋白：表面正电残基中和核酸骨架负电是正常现象 | 正电聚集更宽容（charge_cluster 8、salt_bridge 6、same_sign 5） |
| `membrane` | 膜蛋白：疏水核心严格禁带电 | 核心规则收紧（charge_count 2）、惩罚加重（strength −2.0） |
| `acidic` | 酸性环境（如溶酶体 pH≈5） | 与 default 相同（含 `ph_hint: 5.0` 元数据，供调用方参考） |

> 修改方式：直接编辑 YAML。注意 `structure_aware_filter.py` 的 `default_config()` 也维护一份默认值，若改 default 预设需同步（或让调用统一走 `load_preset`）。

---

## 二、条件默认配置（`code/configs/condition_defaults.yaml`，**Phase 2**）

条件向量 shape `[7]`（mask-aware，顺序固定）：
```
[pH, has_charge_flag, charge_val, has_pos_limit_flag, pos_limit_val, has_neg_limit_flag, neg_limit_val]
```
`has_X_flag`（0/1）告诉网络哪些值是真实条件、哪些是占位符，避免 0 值歧义。

| 键 | 含义 | 当前值 |
|----|------|--------|
| `cond_dim` | 条件向量维度 | 7 |
| `pH_min` / `pH_max` | 训练时连续采样 pH 范围 | 4.0 / 10.0 |
| `default_net_charge` | 可选目标净电荷默认值 | null |
| `default_local_pos_limit` / `default_local_neg_limit` | 10Å 内正/负电荷数上限 | 6 / 6 |
| `normalization.mean` / `.std` | 每维标准化常量（训练前从训练集统计后填入） | **null（训练前必填）** |
| `encoder.hidden_dim` / `token_dim` / `n_tokens` | 条件编码器结构 | 64 / 128 / 4 |

⚠️ **训练前必须完成**：从训练集逐维度算 μ/σ 填入 `normalization`（不同量纲的条件直接进 MLP 会梯度不稳，`docs/TECH.md` §3.2 讨论过 softmax 问题，这里是输入量纲问题）。

---

## 三、命令行参数（`run_guided.py`）

| 参数 | 类型 | 默认 | 含义 | 建议 |
|------|------|------|------|------|
| `--pdb` | 路径 | 必填 | 输入 PDB 文件（多链自动取蛋白链） | 纯蛋白链 |
| `--pH` | float | 必填 | 工作环境 pH | 4–10（pKa 表覆盖范围） |
| `--target_charge` | float | None | 目标净电荷；None=不引导电荷，只做结构过滤 | 先算 native 电荷做参考 |
| `--preset` | str | default | 结构过滤器场景预设 | default/nucleic_acid_binding/membrane/acidic |
| `--num_samples` | int | 10 | 生成候选序列数 | 5–20 |
| `--temperature` | float | 0.3 | 采样温度（低=更保守） | 0.1–0.5 |
| `--strength` | float | 0.5 | 电荷引导强度（大=更强但破坏模型先验） | 0.2–0.5 |
| `--seed` | int | 111 | 随机种子（可复现） | 任意 |
| `--weights` | 路径 | **MoMPNN 默认权重** | 模型权重：MoMPNN 的 `.ckpt`（默认）；LigandMPNN 的 `.pt`（显式回退） | 见下文 |
| `--model_type` | str | auto | auto=按权重自动检测；protein_mpnn=纯 backbone（MoMPNN）；ligand_mpnn=配体上下文（原版） | 一般用 auto |
| `--out_dir` | 路径 | 自动 | 输出目录（默认 `output/guided_<pdb>_pH<pH>/`） | 建议显式指定 |

**权重说明**：
- **默认 MoMPNN**（E4 决策）：`MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt`（多目标 DPO 微调版，纯 backbone，`--model_type auto` 自动识别为 protein_mpnn）
- 显式回退原版 LigandMPNN（含配体上下文）：`LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt`

**自动检测逻辑**：权重 ckpt 里有 `atom_context_num`（>0）→ ligand_mpnn；没有 → protein_mpnn。

---

## 四、环境配置

| 项 | 值 | 说明 |
|----|-----|------|
| 主环境 | `confumpnn`（Python 3.11, torch 2.2.1+cu121） | Phase 1 推理/开发 |
| ESMFold | `confumpnn-esmfold`（torch 2.6.0+cu124, fair-esm 2.0.0） | 回折验证（pLDDT/TM-score） |
| TemBERTure | `confumpnn-temberture`（torch 2.13 CPU） | 热稳定打分 |
| Protein-Sol | 系统 python + Perl 5.34 | 可溶打分（`protein_sol_mcp/`） |

> 环境细节与踩坑见 memory `confumpnn-env-setup.md` 与项目根 `CLAUDE.md`。

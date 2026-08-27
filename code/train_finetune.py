"""Phase 2 条件微调训练脚本 —— 让模型真正「感知」pH 电荷约束。

背景
----
Phase 1 的诚实边界：纯 logit bias 引导下，模型自身没有任何 pH 先验
（同一蛋白各 pH 序列完全相同，电荷差异全靠 net_charge 物理计算）。
本脚本是「模型 pH 感知」的正解：条件微调。

架构（与 PROJECT_PLAN.md 4.5 / PROJECT_EXTEND.md 一致）
-----------------------------------------------------
    backbone        = MoMPNN（E4 默认生成器，ProtAlign 多目标 DPO 权重）——【冻结】
    ConditionEncoder = Soft Prompt MLP（condition_embedding.py）——【唯一可训练 ~75K 参数】

soft prompt 注入（对 4.5 的实现修正）
    PROJECT_PLAN 写「4 个 token 拼到 decoder 输入前缀」。但 decoder 的 E_idx / order mask
    依赖固定 L 个位置，字面前缀需重排 E_idx，易错。实际实现为 **cross-attention**：
        h_V += softmax(h_V · prompt^T / √d) · prompt          # [B,L,128] + [B,L,128]
    每个结构节点按需读取 4 个条件 token，等价于 soft prompt，且无需改动解码器。

损失
----
    L = CE + λ_c·charge_deviation + λ_kl·KL(条件化 ‖ 无条件) + λ_keep·SeqKeep

    CE                : 重建 native 序列（结构匹配度锚，PROJECT_PLAN 6.1 风险表）
    charge_deviation  : 期望净电荷 vs 目标电荷（可微，differentiable_charge.py）
    KL-anchor（防失控）: 约束条件注入后的输出分布不偏离 backbone 无条件输出太远。
                        只允许在电荷约束要求的反向上变化 → 防止微调破坏 MoMPNN 的
                        可溶/Tm/可设计性（这些目标存在冻结的 backbone 权重里）。
                        （溶解性/Tm 属第二版多目标微调，不是本阶段损失；本阶段
                        防护 = 冻结 backbone + CE 锚 + KL 锚 + 事后 E1b 验证。）
    SeqKeep（序列保持，治 S1 注入选择性，第十四轮新增）:
                        以「无条件 argmax 序列」为锚，对**自洽样本**（target=native）
                        做 CE——条件输出逐位逼近无条件输出。这是判断标准 v1 的 S1
                        判据（A 场景条件臂 vs 基线 identity ≥ 0.7）的训练侧直接对应，
                        比 KL 更直接（KL 管分布距离，管不住 argmax 翻盘）。
                        **只在自洽样本施加**；扰动样本 target≠native 时电荷偏移是
                        期望行为，不受此正则约束。

数据 / 目标（自洽 + 扰动混合）
------------------------------
    样本 = (骨架, native 序列, 条件向量[7])，来自 data/cath/labels.npz
    （999 结构域 × 8 pH = 7992 样本；每域批内共享结构，仅条件向量不同 → 批内 B=8）

    目标电荷的策略（关键）：
      若目标恒等于 native 电荷，CE 与电荷损失同时被「重建 native」满足，
      模型学不到电荷偏移能力（条件向量变成无效输入）。
      因此用 **混合目标**：
        · 70%：自洽目标 = native 序列在该 pH 的净电荷（锚定结构，label 原值）
        · 30%：扰动目标 = native 电荷 ± Uniform[1, perturb_scale]
          ——制造 CE 与电荷损失的冲突，教模型「target 偏离 native 时如何偏移
            氨基酸分布」。这正是计划 Go/No-Go（pH↑→偏负电残基增多）的学习信号。
      ⚠️ 第十四轮修正：扰动比例从 50% 降到 30%（原生标签 50%→70%），
         让「target=原生时保持」成为主导训练信号；配合 SeqKeep 正则治 S1。

    λ_c / λ_kl / λ_keep / perturb_prob / perturb_scale 均可命令行调节。

用法
----
    conda activate confumpnn
    cd /data/nfs/IC/baokun_yu/ConfuMPNN/code
    python train_finetune.py --device cuda:1 \
      --epochs 30 --out_dir ../output/finetune

后台运行（端口重置/退出终端不中断）：
    nohup setsid /home/baokun_yu/miniconda3/envs/confumpnn/bin/python \
      code/train_finetune.py --device cuda:1 --epochs 30 \
      > code/log/train.log 2>&1 &
    进度查询：bash code/tests/train_status.sh
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CODE_DIR.parent
_LIG_DIR = _ROOT_DIR / "LigandMPNN"
for p in [str(_CODE_DIR), str(_LIG_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data_utils import featurize, parse_PDB  # noqa: E402
from model_utils import ProteinMPNN, cat_neighbors_nodes  # noqa: E402

from src.condition_embedding import ConditionEncoder, make_condition_vector  # noqa: E402
from src.conditioned_sampler import inject_prompt  # noqa: E402  (训练/推理共用同一注入机制)
from src.differentiable_charge import net_charge  # noqa: E402
from src.losses import (  # noqa: E402
    charge_deviation_loss, cross_entropy_loss, sequence_keep_loss,
)
from src.v10_losses import (  # noqa: E402
    ph_aware_structure_penalty, surface_add_charge_loss,
)

# 默认生成器权重（与 run_guided.py E4 一致）
_DEFAULT_WEIGHTS = (
    _ROOT_DIR / "MoMPNN" / "mompnn_paper_checkpoints"
    / "mompnn_temberture_tm_esm_6_4_4_b01.ckpt"
)
_DEFAULT_LABELS = _ROOT_DIR / "data" / "cath" / "labels.npz"
_DEFAULT_DOMPDB = _ROOT_DIR / "data" / "cath" / "S40" / "dompdb"
_DEFAULT_CFG = _CODE_DIR / "configs" / "condition_defaults.yaml"


def parse_args():
    p = argparse.ArgumentParser(description="ConfuMPNN Phase 2 条件微调")
    p.add_argument("--weights", default=str(_DEFAULT_WEIGHTS))
    p.add_argument("--labels", default=str(_DEFAULT_LABELS))
    p.add_argument("--dompdb", default=str(_DEFAULT_DOMPDB))
    p.add_argument("--cfg", default=str(_DEFAULT_CFG))
    p.add_argument("--ligand", action="store_true",
                   help="配体模式（v9）：用 LigandMPNN 权重 + 配体原子上下文特征化")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_c", type=float, default=0.5,
                   help="电荷偏差损失权重")
    p.add_argument("--lambda_kl", type=float, default=0.05,
                   help="KL 锚定正则权重（防失控：约束条件化输出不偏离 backbone）")
    p.add_argument("--lambda_keep", type=float, default=0.5,
                   help="序列保持正则权重（治 S1：自洽样本条件输出逼近无条件 argmax，"
                        "比 KL 更直接地管住 argmax 翻盘）")
    p.add_argument("--perturb_prob", type=float, default=0.3,
                   help="使用扰动电荷目标的概率（制造电荷偏移学习信号；"
                        "第十四轮 0.5→0.3 = 原生标签 70%，治 S1 注入选择性）")
    p.add_argument("--perturb_scale", type=float, default=4.0,
                   help="扰动电荷偏移幅度上限（±Uniform[1,scale]）")
    p.add_argument("--curriculum", action="store_true",
                   help="课程学习（v7）：perturb_scale 从起点随 epoch 渐进到 "
                        "curriculum_scale_max——先学温和偏移，再学极端外推")
    p.add_argument("--curriculum_scale_max", type=float, default=8.0,
                   help="课程学习终点扰动幅度上限")
    p.add_argument("--placeholder_prob", type=float, default=0.15,
                   help="占位符样本比例（从自洽样本中抽取）：把条件电荷置为「不控制」占位，"
                        "让模型学会未指定电荷时的行为（目标 2 的占位符语义）。"
                        "占位两种各半：① has_charge=0 + 值0；② has_charge=1 + 值=训练均值。"
                        "此类样本跳过电荷损失（无 target）")
    p.add_argument("--charge_temp", type=float, default=0.5,
                   help="电荷损失的 softmax 温度（<1 锐化：训练优化的分布≈推理采样分布，"
                        "减小 ~2.57× 过冲；1.0=原版期望电荷）")
    p.add_argument("--loss_reweight", type=int, default=0,
                   help="逆密度加权电荷损失（第十八轮，治高正电 target 外推过冲）："
                        "charge loss 按 target 密度逆加权 weight=k/(density_norm+eps)，"
                        "稀有 target（高正电）权重更大。1=开 0=关")
    p.add_argument("--reweight_k", type=float, default=1.0, help="逆加权基准权重")
    p.add_argument("--reweight_eps", type=float, default=1e-3, help="逆加权分母保护项")
    p.add_argument("--reweight_cap", type=float, default=5.0,
                   help="weight 上限（防高正电样本权重过大压坏负电命中，文献警示 naive 加权过校正）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_domains", type=int, default=0,
                   help="最多用前 N 个结构域（0=全部；冒烟测试用）")

    # ---- v10 三组件（v3 §3.1）----
    p.add_argument("--decouple_perturb", action="store_true",
                   help="v10 A 条件解耦：扰动 target 与骨架 native 无关（Uniform[-range,range]），"
                        "打破'骨架类型与 target 电荷强耦合'（碱性骨架只见正电 target → 只能外推）")
    p.add_argument("--decouple_range", type=float, default=12.0,
                   help="A 的随机 target 范围（±Uniform[-range, range]，v3 预期扩大可靠区至 "
                        "[native−10, native+10]）")
    p.add_argument("--add_supervision", action="store_true",
                   help="v10 B 表面添加电荷监督：L_add 直接对抗'只删不加'——需要更负→表面加 D/E、"
                        "更正要加 K/R，只动表面（fracSASA≥θ），以净电荷目标为上界")
    p.add_argument("--lambda_add", type=float, default=0.3,
                   help="B 的 L_add 权重（v3 建议扫 0.1/0.3/0.5）")
    p.add_argument("--sasa_threshold", type=float, default=0.25,
                   help="B 表面资格门槛 θ（fracSASA ≥ θ 才计入 L_add）")
    p.add_argument("--ph_aware_filter", action="store_true",
                   help="v10 C 结构惩罚增强：训练时用 P0-5 的 pH 自适应过滤器 bias 压制电荷聚集"
                        "（His/Cys/Tyr 按质子化态纳入），大额添加样本动态加强（scale_boost）")
    p.add_argument("--structure_boost", type=float, default=1.5,
                   help="C 对大额添加扰动样本的结构惩罚放大系数（scale_boost）")

    p.add_argument("--out_dir", default=str(_CODE_DIR / "output" / "finetune"))
    p.add_argument("--log_progress", default=str(_CODE_DIR / "log" / "train_progress.json"))
    p.add_argument("--log_file", default=str(_CODE_DIR / "log" / "train.log"))
    return p.parse_args()


def build_density_table(charge_arr, perturb_scale, lo=-40.0, hi=60.0, bw=0.5):
    """估算训练时 target 电荷的分布密度（native charge@各pH + ±扰动扩展）。

    用于逆密度加权（--loss_reweight）：charge loss 按 target 密度逆加权，
    稀有 target（高正电，训练分布尾部）权重更大。文献：不均衡回归逆密度加权
    （US11720818B2 / arXiv 2506.01486）。

    返回 (density_norm, bucket)：归一化密度表（0~1，最密=1）+ 分箱索引函数。
    """
    n_buckets = int((hi - lo) / bw)
    counts = np.zeros(n_buckets, dtype=np.float64)

    def bucket(c):
        return int(np.clip((c - lo) / bw, 0, n_buckets - 1))

    for c in charge_arr:
        c0, c1 = bucket(c - perturb_scale), bucket(c + perturb_scale)
        counts[c0:c1 + 1] += 1.0   # 扰动对称均匀，target 近似均匀落在区间内
    counts = np.maximum(counts, 1.0)
    density = counts / counts.sum()
    density_norm = density / density.max()   # [0,1]，最密箱=1 → weight 中间≈k
    return density_norm, bucket


def load_backbone(weights, device, ligand=False):
    """加载 backbone。MoMPNN=纯 backbone ProteinMPNN；--ligand 时用 LigandMPNN 权重。

    --ligand 时自动检测权重类型（同 run_guided.py load_model）：
    权重含 atom_context_num（>0）→ ligand_mpnn（配体上下文）；否则 protein_mpnn。
    """
    checkpoint = torch.load(weights, map_location=device)
    if ligand:
        model_type = (
            "ligand_mpnn" if checkpoint.get("atom_context_num", 0) > 0
            else "protein_mpnn"
        )
        atom_context_num = (
            0 if model_type == "protein_mpnn"
            else int(checkpoint.get("atom_context_num", 16))
        )
    else:
        model_type = "protein_mpnn"
        atom_context_num = 0
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=int(checkpoint["num_edges"]),
        device=device,
        atom_context_num=atom_context_num,
        model_type=model_type,
        ligand_mpnn_use_side_chain_context=0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_domain(feature_dict, device, seed):
    """把 parse_PDB + featurize 的结果整理成训练需要的张量。

    返回 dict（全部在 device 上）：
        X, S, mask, chain_mask, R_idx, chain_labels, randn
    + 透传 featurize 全部键（ligand 模式的 Y/Y_t/Y_m/mask_XY 供 backbone.encode 用）。
    其中 randn 用 seed 固定（每域一个解码顺序，批内 8 个 pH 共享）。
    """
    L = feature_dict["X"].shape[1]
    fd = {k: v.to(device) if torch.is_tensor(v) else v for k, v in feature_dict.items()}
    rng = np.random.RandomState(seed)
    randn = torch.from_numpy(rng.randn(L).astype(np.float32)).to(device)
    dom = dict(fd)  # 透传全部键（含配体上下文）
    dom.update({
        "X": fd["X"], "S": fd["S"], "mask": fd["mask"].float(),
        "chain_mask": fd["chain_mask"].float(),
        "R_idx": fd["R_idx"], "chain_labels": fd["chain_labels"],
        "randn": randn,
    })
    return dom


def decoder_forward(model, h_V, h_E, E_idx, dom, B, device):
    """Teacher-forced 并行解码（ProteinMPNN 标准训练前向）。

    每个位置按 decoding_order 只关注早于它的位置（mask_bw），h_S = W_s(S_true)。
    返回 logits [B, L, 21]。
    """
    S_true = dom["S"].long().repeat(B, 1)
    mask = dom["mask"].repeat(B, 1)
    chain_mask = dom["chain_mask"].repeat(B, 1)
    randn = dom["randn"].repeat(B, 1)
    L = S_true.shape[1]

    decoding_order = torch.argsort((chain_mask + 0.0001) * torch.abs(randn))  # [B,L]
    permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=L).float()
    order_mask_backward = torch.einsum(
        "ij, biq, bjp->bqp",
        (1 - torch.triu(torch.ones(L, L, device=device))),
        permutation_matrix_reverse,
        permutation_matrix_reverse,
    )  # [B, L, L]
    mask_attend = torch.gather(order_mask_backward, 2, E_idx.repeat(B, 1, 1)).unsqueeze(-1)
    mask_1D = mask.view(B, L, 1, 1)
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1.0 - mask_attend)

    h_S = model.W_s(S_true)  # [B, L, 128]
    h_ES = cat_neighbors_nodes(h_S, h_E.repeat(B, 1, 1, 1), E_idx.repeat(B, 1, 1))
    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E.repeat(B, 1, 1, 1),
                                       E_idx.repeat(B, 1, 1))
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx.repeat(B, 1, 1))
    h_EXV_encoder_fw = mask_fw * h_EXV_encoder

    for layer in model.decoder_layers:
        h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx.repeat(B, 1, 1))
        h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
        h_V = layer(h_V, h_ESV, mask)

    return model.W_out(h_V)  # [B, L, 21]


def kl_anchor_loss(logits, logits_ref, mask):
    """KL(p_ref ‖ p_cond)：条件化输出分布不偏离 backbone 无条件分布太远。

    对每个位置求 KL，再按 mask 平均。p_ref 冻结（常数项），梯度只流向条件化分支。
    """
    p_ref = F.softmax(logits_ref, dim=-1)
    logp_cond = F.log_softmax(logits, dim=-1)
    kl = (p_ref * (p_ref.log() - logp_cond)).sum(dim=-1)  # [B, L]
    denom = mask.float().sum().clamp(min=1.0)
    return (kl * mask.float()).sum() / denom


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = open(args.log_file, "a")
    def logln(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    logln(f"=== ConfuMPNN Phase 2 条件微调启动（v10）===")
    logln(f"device={device}  epochs={args.epochs}  lr={args.lr}  "
          f"λ_c={args.lambda_c}  λ_kl={args.lambda_kl}  λ_keep={args.lambda_keep}  "
          f"perturb_prob={args.perturb_prob}  perturb_scale={args.perturb_scale}  "
          f"placeholder_prob={args.placeholder_prob}  "
          f"charge_temp={args.charge_temp}")
    if args.decouple_perturb:
        logln(f"[v10 A] 条件解耦开：target 与 native 无关 Uniform[-{args.decouple_range},"
              f"{args.decouple_range}]")
    if args.add_supervision:
        logln(f"[v10 B] 表面添加电荷监督开：λ_add={args.lambda_add}  SASA θ={args.sasa_threshold}")
    if args.ph_aware_filter:
        logln(f"[v10 C] pH 自适应结构惩罚开：boost={args.structure_boost}")

    # ---- backbone + 条件编码器 ----
    backbone = load_backbone(args.weights, device, ligand=args.ligand)
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()
    n_backbone = sum(p.numel() for p in backbone.parameters())
    logln(f"backbone: MoMPNN 冻结（{n_backbone/1e6:.2f}M 参数，不更新）")

    import yaml
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)["condition_defaults"]
    enc = ConditionEncoder(
        cond_dim=cfg["cond_dim"],
        hidden_dim=cfg["encoder"]["hidden_dim"],
        token_dim=cfg["encoder"]["token_dim"],
        n_tokens=cfg["encoder"]["n_tokens"],
        mean=cfg["normalization"]["mean"],
        std=cfg["normalization"]["std"],
    ).to(device)
    n_trainable = sum(p.numel() for p in enc.parameters())
    logln(f"ConditionEncoder 可训练（{n_trainable} 参数，唯一更新对象）")

    optimizer = torch.optim.Adam(enc.parameters(), lr=args.lr)

    # ---- 数据：labels.npz + dompdb ----
    labels = np.load(args.labels, allow_pickle=True)
    domain_ids = labels["domain_ids"]            # [999]
    pH_arr = labels["pH"].astype(np.float32)     # [7992] 域主序
    charge_arr = labels["charge"].astype(np.float32)  # [7992]
    n_dom = len(domain_ids)
    if args.max_domains > 0:
        n_dom = min(n_dom, args.max_domains)
    n_pH = pH_arr.size // len(domain_ids)       # 每域 pH 数（=8）
    assert n_pH * len(domain_ids) == pH_arr.size
    logln(f"数据：{n_dom} 结构域 × {n_pH} pH = {n_dom*n_pH} 样本（{args.labels}）")

    # 逆密度加权：预计算训练 target 密度表（--loss_reweight，第十八轮）
    density_norm, density_bucket = None, None
    if args.loss_reweight:
        density_norm, density_bucket = build_density_table(charge_arr, args.perturb_scale)
        logln(f"逆密度加权开：k={args.reweight_k} eps={args.reweight_eps} "
              f"cap={args.reweight_cap}（电荷分布偏斜补偿，治高正电 target 过冲）")

    # 预解析每个结构域 → 缓存 feature 张量 + encode + 无条件 logits
    # （backbone 冻结 → encode 与无条件输出每域只需算一次，全 epoch 复用）
    # prody parsePDB 按文件后缀判断格式：无 .pdb 后缀会被当 mmCIF 解析。
    # CATH 文件无扩展名 → 用 .pdb 后缀的符号链接目录（data/ 下，git 不跟踪）。
    # 配体模式（v9）：domain_ids 含真实文件名（如 1JCG.pdb/1abc.cif），
    # 直接从 --dompdb 定位真实文件，跳过 symlink（后缀真实，prody 可解析）。
    dom_cache_dir = Path(args.dompdb).parent / (Path(args.dompdb).name + "_pdb")
    dom_cache_dir.mkdir(exist_ok=True)
    abs_dompdb = os.path.abspath(args.dompdb)

    domains = []
    n_ok, n_skip = 0, 0
    for i, did in enumerate(domain_ids[:n_dom]):
        # 优先：真实文件已带后缀（配体数据 .pdb/.cif，prody 按后缀判格式）。
        # ⚠️ CATH 域 id 无后缀（如 3t97A00）——direct.exists() 为 True 但 prody
        # 会把无后缀当 mmCIF 解析失败。因此只有「带后缀」才直用，否则走 .pdb symlink。
        direct = Path(args.dompdb) / str(did)
        has_suffix = Path(str(did)).suffix in (".pdb", ".cif", ".ent", ".cif.gz", ".pdb.gz")
        # ⚠️ 无后缀的 CATH 域走 .pdb symlink；symlink 创建放在 try 内，
        #    目标文件缺失的域（dangling link）会被当作坏域跳过，不崩溃。
        try:
            if direct.exists() and has_suffix:
                pdb_path = str(direct)
            else:
                link_path = dom_cache_dir / f"{did}.pdb"
                # lexists 检查 symlink 本身存在（避免 dangling link 上 Path.exists()=False
                # 导致 os.symlink 报 FileExistsError）；目标是绝对路径，与 cwd 无关。
                if not os.path.lexists(link_path):
                    os.symlink(os.path.join(abs_dompdb, str(did)), link_path)
                pdb_path = str(link_path)
            protein_dict, *_ = parse_PDB(pdb_path, device="cpu", parse_all_atoms=False)
            L = protein_dict["X"].shape[0]
            # 单链 CATH 域：全部残基设计 → chain_mask = 全 1
            protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
            feature_dict = featurize(
                protein_dict,
                use_atom_context=args.ligand,
                number_of_ligand_atoms=(16 if args.ligand else 0),
                model_type=("ligand_mpnn" if args.ligand else "protein_mpnn"),
            )
            dom = build_domain(feature_dict, device, seed=args.seed + i)
            # 冻结 backbone 上一次性 encode
            h_V, h_E, E_idx = backbone.encode(dom)
            dom["h_V"] = h_V
            dom["h_E"] = h_E
            dom["E_idx"] = E_idx
            # 无条件 logits（无 prompt 注入）——KL 锚参考分布，每域一次
            with torch.no_grad():
                logits_uncond = decoder_forward(backbone, h_V, h_E, E_idx, dom, 1, device)
            dom["logits_uncond"] = logits_uncond
            # 无条件 argmax 序列（SeqKeep 锚，常数）：X 位置锚到 0，靠 ce_mask 排除。
            # ⚠️ dom["S"] 带 batch 维 [1,L]，索引 [0] 取单链，避免广播成 [1,L]。
            anchor = logits_uncond[0].argmax(-1)                          # [L]
            anchor = torch.where(dom["S"][0] < 20, anchor, torch.zeros_like(anchor))  # [L]
            dom["seq_anchor"] = anchor
            # CE 有效性掩码：排除非标准残基（S==20 的 X）
            valid = (dom["S"] < 20).float()
            dom["ce_mask"] = dom["mask"] * dom["chain_mask"] * valid
            # v10 B 表面添加电荷监督：逐域预计算 fractional SASA（freesasa，backbone 冻结
            # → 结构静态 → SASA 每域只需算一次）。供 L_add 的"只加表面"权重。
            # ⚠️ freesasa(Bio.PDB) 与 LigandMPNN parse_PDB 对残基判定可能不同（freesasa
            # 可能多出 parse 跳过的残基，如链端异常残基 136-138）。**正确对齐 = resnum 交集**：
            #   sasa 的 residues[] 与 dom 的 R_idx[] 做集合匹配，只取两边都有的残基号，
            #   freesasa 多出的残基忽略，parse 独有的残基（若有）frac 补 0。
            #   → 不再有"长度不匹配"，L_add 覆盖全部对齐残基（用 X/非标准位置 frac=0）。
            if args.add_supervision:
                try:
                    from src.sasa import fractional_sasa
                    sasa_info = fractional_sasa(pdb_path,
                                                surface_threshold=args.sasa_threshold,
                                                align_to_full=False)  # 只返回标准 AA 位置
                    sasa_frac = sasa_info["frac_sasa"]     # [n_sasa]
                    sasa_resids = sasa_info["residues"]    # [n_sasa] 残基号
                    # dom["R_idx"] 形状 [1, L]（featurize 带 batch 维），展平到 [L]
                    dom_resids = np.asarray(dom["R_idx"].cpu().numpy()).reshape(-1)
                    # resnum 交集映射：sasa 残基号 → 索引
                    sasa_map = {int(r): i for i, r in enumerate(sasa_resids)}
                    aligned = np.zeros(L, dtype=np.float64)
                    n_aligned = 0
                    for pos in range(L):
                        rid = int(dom_resids[pos])
                        if rid in sasa_map:
                            aligned[pos] = sasa_frac[sasa_map[rid]]
                            n_aligned += 1
                        # 不在 sasa（如 parse 独有/非标准）→ 保持 0（埋藏/不参与）
                    dom["frac_sasa"] = aligned
                    if n_aligned < L:
                        logln(f"  ℹ️ {did} resnum 对齐 {n_aligned}/{L}（{L-n_aligned} 个非标准/parse独有残基 frac=0）")
                except Exception as e:
                    dom["frac_sasa"] = None
                    logln(f"  ⚠️ {did} SASA 计算失败: {e}（跳过 L_add）")
            # 该域 8 个 (pH, charge) 条件（按已接受域数索引，跳过坏域不错位）
            idx0 = n_ok * n_pH
            dom["pH"] = pH_arr[idx0:idx0 + n_pH]
            dom["charge_label"] = charge_arr[idx0:idx0 + n_pH]
            domains.append(dom)
            n_ok += 1
        except Exception as e:
            n_skip += 1
            logln(f"  ⚠️ 跳过坏域 {did}: {e}")
        if (i + 1) % 200 == 0:
            logln(f"  预解析+encode {n_ok}/{n_dom}（跳过 {n_skip}）")
    if n_skip:
        logln(f"⚠️ 共跳过 {n_skip} 个坏域（prody 无法解析），实际训练 {n_ok} 域")

    # 记录每域梯度归零的分界（backbone 参数全 requires_grad=False，
    # 只传 enc.parameters() 给 optimizer，天然只更新编码器）
    total_cached = sum(d["h_E"].numel() for d in domains) * 4 / 1e9
    logln(f"预解析完成，缓存 encode 特征 ~{total_cached:.2f}GB")

    # ---- 训练 ----
    n_dom_eff = len(domains)  # 实际可训练域数（跳过坏域后可能 < n_dom）
    domain_idx = list(range(n_dom_eff))
    n_steps_total = args.epochs * n_dom_eff
    step = 0
    t_start = time.time()
    logln(f"开始训练：{args.epochs} epochs × {n_dom_eff} 域/epoch = {n_steps_total} steps")

    for epoch in range(1, args.epochs + 1):
        # v7 课程学习：扰动幅度从 perturb_scale 随 epoch 线性渐进到 curriculum_scale_max
        if args.curriculum:
            progress = (epoch - 1) / max(args.epochs - 1, 1)   # 0→1
            scale_cur = args.perturb_scale + (
                args.curriculum_scale_max - args.perturb_scale) * progress
            if epoch == 1:
                logln(f"课程学习开：perturb_scale {args.perturb_scale} → "
                      f"{args.curriculum_scale_max}（共 {args.epochs} epochs）")
        else:
            scale_cur = args.perturb_scale
        random.shuffle(domain_idx)
        epoch_loss, epoch_ce, epoch_cd, epoch_kl, epoch_keep = [], [], [], [], []
        # charge loss 分拆监控（v7 归因）：self=自洽/占位, mild=温和扰动, extreme=极端扰动(|offset|≥5)
        grp_cd = {"self": [], "mild": [], "extreme": []}
        for di in domain_idx:
            dom = domains[di]
            B = n_pH
            # 条件向量 [B, 7]
            pH_b = torch.from_numpy(dom["pH"]).to(device)              # [8]
            charge_b = torch.from_numpy(dom["charge_label"]).to(device)  # [8]
            # 混合目标：70% 自洽（target=native）+ 30% 扰动电荷（制造偏移学习信号）
            mask_p = torch.zeros(B, dtype=torch.bool, device=device)
            if args.perturb_prob > 0:
                mask_p = (torch.rand(B, device=device) < args.perturb_prob)
                # v10 A 条件解耦（--decouple_perturb）：扰动 target 与骨架 native **无关**——
                # 直接采样 Uniform[-decouple_range, decouple_range] 的独立随机 target，
                # 打破"碱性骨架只见正电 target / 中性骨架只见温和 target"的耦合，
                # 让"中性骨架+高正电"等组合进入训练分布（v3 §3.1 A）。
                # 关闭时（默认）沿用 v7/v9：native ± Uniform[1, scale]（幅度受课程控制）。
                if args.decouple_perturb:
                    offset = torch.where(
                        mask_p,
                        (torch.rand(B, device=device) * 2 - 1) * args.decouple_range,
                        torch.zeros(B, device=device),
                    )
                else:
                    offset = torch.where(
                        mask_p,
                        torch.randint(1, int(scale_cur) + 1, (B,), device=device).float()
                        * torch.where(torch.rand(B, device=device) < 0.5, 1.0, -1.0),
                        torch.zeros(B, device=device),
                    )
                charge_b = charge_b + offset
            # 占位符样本（目标 2：部分条件不控制）：从自洽样本中随机抽取，把电荷条件置为占位。
            # 第十七轮修正：统一用"均值占位"（has_charge=1 + 值=训练均值 −1.34），符合目标 2
            # "部分条件不控制用非 0 占位符替代"；并施加电荷损失（目标=均值），让"占位"落在
            # 温和可折叠默认。第十六轮实证：flag=0+值0 语义因无电荷监督 + seq-keep 锚定无条件
            # argmax（负漂移基线）→ 推理占位时电荷极端负极化（−8~−16）→ 折叠全失败。
            mask_ph = torch.zeros(B, dtype=torch.bool, device=device)
            if args.placeholder_prob > 0:
                mask_ph = (~mask_p) & (torch.rand(B, device=device) < args.placeholder_prob)
            charge_mean = float(cfg["normalization"]["mean"][2])  # 电荷维度训练均值
            cond_b = torch.stack([
                make_condition_vector(p, c) if not mask_ph[i] else
                make_condition_vector(p, net_charge=charge_mean)
                for i, (p, c) in enumerate(zip(pH_b.tolist(), charge_b.tolist()))
            ]).to(device)  # [8, 7]

            # 条件注入 + 解码
            prompt = enc(cond_b)                 # [8, 4, 128]
            h_V = dom["h_V"].repeat(B, 1, 1)
            h_V = inject_prompt(h_V, prompt)
            logits = decoder_forward(backbone, h_V, dom["h_E"], dom["E_idx"], dom, B, device)

            S_true = dom["S"].long().repeat(B, 1)
            ce_mask = dom["ce_mask"].repeat(B, 1)
            ce = cross_entropy_loss(logits, S_true, ce_mask)

            # 电荷偏差（逐样本，因 pH 每样本不同；温度化：优化采样分布电荷而非期望电荷）
            # 第十七轮：占位符样本也施加电荷损失（target=训练均值）——"占位"=温和默认电荷，
            # 而非完全无监督（第十六轮实证无监督→负漂移→折叠失败）
            cd = torch.zeros(B, device=device)
            for i in range(B):
                tgt_i = charge_mean if mask_ph[i] else charge_b[i]
                cd[i] = charge_deviation_loss(
                    logits[i:i+1], pH=pH_b[i], target_charge=tgt_i,
                    mask=ce_mask[i:i+1], temperature=args.charge_temp,
                )
                if args.loss_reweight:
                    w = args.reweight_k / (
                        density_norm[density_bucket(float(tgt_i))] + args.reweight_eps)
                    cd[i] *= min(w, args.reweight_cap)
                # v7 分拆监控：self=自洽/占位, mild=温和扰动, extreme=极端扰动
                if mask_ph[i] or not mask_p[i]:
                    grp_cd["self"].append(cd[i].item())
                elif abs(offset[i].item()) >= 5:
                    grp_cd["extreme"].append(cd[i].item())
                else:
                    grp_cd["mild"].append(cd[i].item())
            cd = cd.mean()

            # KL 锚（条件化 ‖ 无条件）
            ref = dom["logits_uncond"].repeat(B, 1, 1)
            kl = kl_anchor_loss(logits, ref, ce_mask) if args.lambda_kl > 0 else torch.zeros((), device=device)

            # 序列保持正则（仅自洽样本：未扰动 → target=native → 条件输出逼近无条件 argmax；
            # 扰动样本 target≠native，电荷偏移是期望行为，不受约束）
            keep = torch.zeros(B, device=device)
            if args.lambda_keep > 0:
                anchor = dom["seq_anchor"].unsqueeze(0)  # [1, L]
                for i in range(B):
                    if not mask_p[i].item():
                        keep[i] = sequence_keep_loss(
                            logits[i:i+1], anchor, ce_mask[i:i+1])
            keep = keep.mean()

            # ---- v10 B 表面添加电荷监督（--add_supervision）：对抗"只删不加" ----
            # 只在扰动样本上施加（自洽 target=native 无"需要添加电荷"的需求）。
            # 需要的电荷增量 = offset[i]（扰动相对 native 的偏移方向）：
            #   需要更负（offset<0）→ 表面加 D/E；需要更正（offset>0）→ 表面加 K/R。
            add = torch.zeros(B, device=device)
            n_add = 0
            if args.add_supervision and dom.get("frac_sasa") is not None:
                for i in range(B):
                    if mask_ph[i].item() or not mask_p[i].item():
                        continue  # 只对扰动样本施加
                    delta = float(offset[i].item())
                    if abs(delta) < 1.0:
                        continue  # 需求过小，不启用
                    add[i] = surface_add_charge_loss(
                        logits[i:i+1], dom["frac_sasa"],
                        target_surface_charge_delta=delta,
                        surface_threshold=args.sasa_threshold,
                    )
                    n_add += 1
            add = add.mean() if n_add else torch.zeros((), device=device)

            # ---- v10 C 结构惩罚增强（--ph_aware_filter）：动态压制电荷聚集 ----
            # 用 P0-5 的 pH 自适应过滤器 bias（His/Cys/Tyr 按质子化态纳入）；
            # 对"大额添加"的扰动样本放大惩罚（scale_boost），防电荷成簇。
            struct_pen = torch.zeros((), device=device)
            if args.ph_aware_filter:
                from src.structure_aware_filter import StructureAwareFilter
                coords = dom["X"][0, :, 1].cpu().numpy()  # [L,3] Cα
                filt = StructureAwareFilter(coords)
                seq_int_cur = dom["S"][0].long().cpu().numpy()
                # 按样本方向决定是否加强（大额扰动 → 加强）
                sp, sp_info = ph_aware_structure_penalty(
                    logits, filt, seq_int_cur, pH=float(pH_b[0].item()),
                    mask=ce_mask, scale_boost=1.0,
                )
                # 对扰动样本（尤其大额）动态加强
                boost = args.structure_boost if mask_p.any().item() else 1.0
                if boost > 1.0:
                    sp2, _ = ph_aware_structure_penalty(
                        logits, filt, seq_int_cur, pH=float(pH_b[0].item()),
                        mask=ce_mask, scale_boost=boost,
                    )
                    struct_pen = sp2
                else:
                    struct_pen = sp

            total = ce + args.lambda_c * cd + args.lambda_kl * kl + args.lambda_keep * keep
            if args.add_supervision:
                total = total + args.lambda_add * add
            if args.ph_aware_filter:
                total = total + 0.05 * struct_pen

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            optimizer.step()

            epoch_loss.append(total.item()); epoch_ce.append(ce.item())
            epoch_cd.append(cd.item()); epoch_kl.append(kl.item())
            epoch_keep.append(keep.item())
            step += 1

        # ---- epoch 汇总 + 进度文件 ----
        avg = lambda x: float(np.mean(x))
        grp_str = "  ".join(f"{k}={avg(v):.3f}" for k, v in grp_cd.items() if v)
        msg = (f"epoch {epoch}/{args.epochs}  total={avg(epoch_loss):.4f}  "
               f"ce={avg(epoch_ce):.4f}  charge={avg(epoch_cd):.4f}  "
               f"kl={avg(epoch_kl):.4f}  keep={avg(epoch_keep):.4f}")
        if grp_str:
            msg += f"  [cd {grp_str}]"
        msg += f"  elapsed={((time.time()-t_start)/60):.1f}min"
        logln(msg)
        prog = {
            "epoch": epoch, "total_epochs": args.epochs,
            "loss": avg(epoch_loss), "ce": avg(epoch_ce),
            "charge": avg(epoch_cd), "kl": avg(epoch_kl),
            "keep": avg(epoch_keep),
            "elapsed_min": round((time.time() - t_start) / 60, 1),
        }
        with open(args.log_progress, "w") as f:
            json.dump(prog, f, indent=2)

        # ---- checkpoint ----
        ckpt_path = out_dir / f"finetune_epoch{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "condition_encoder_state": enc.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cond_dim": cfg["cond_dim"],
            "n_tokens": cfg["encoder"]["n_tokens"],
            "token_dim": cfg["encoder"]["token_dim"],
            "mean": cfg["normalization"]["mean"],
            "std": cfg["normalization"]["std"],
            "backbone_weights": args.weights,
            "loss_terms": prog,
            # 追溯字段（第十四轮修正参数 + 第十五轮占位符）
            "perturb_prob": args.perturb_prob,
            "perturb_scale": args.perturb_scale,
            "placeholder_prob": args.placeholder_prob,
            "lambda_keep": args.lambda_keep,
            "charge_temp": args.charge_temp,
            "curriculum": args.curriculum,
            "curriculum_scale_max": args.curriculum_scale_max,
            # v10 三组件追溯字段（v3 §3.1）
            "decouple_perturb": args.decouple_perturb,
            "decouple_range": args.decouple_range,
            "add_supervision": args.add_supervision,
            "lambda_add": args.lambda_add,
            "sasa_threshold": args.sasa_threshold,
            "ph_aware_filter": args.ph_aware_filter,
            "structure_boost": args.structure_boost,
        }, ckpt_path)
        # 保留最新一份 alias，方便推理时加载
        torch.save(enc.state_dict(), out_dir / "condition_encoder_last.pt")

    logln(f"训练完成。总耗时 {((time.time()-t_start)/60):.1f}min。checkpoint 在 {out_dir}/")
    log.close()


if __name__ == "__main__":
    main()

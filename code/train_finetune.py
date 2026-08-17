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
    p.add_argument("--charge_temp", type=float, default=0.5,
                   help="电荷损失的 softmax 温度（<1 锐化：训练优化的分布≈推理采样分布，"
                        "减小 ~2.57× 过冲；1.0=原版期望电荷）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_domains", type=int, default=0,
                   help="最多用前 N 个结构域（0=全部；冒烟测试用）")
    p.add_argument("--out_dir", default=str(_CODE_DIR / "output" / "finetune"))
    p.add_argument("--log_progress", default=str(_CODE_DIR / "log" / "train_progress.json"))
    p.add_argument("--log_file", default=str(_CODE_DIR / "log" / "train.log"))
    return p.parse_args()


def load_backbone(weights, device):
    """复刻 run_guided.py load_model：MoMPNN = 纯 backbone ProteinMPNN。"""
    checkpoint = torch.load(weights, map_location=device)
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=int(checkpoint["num_edges"]),
        device=device,
        atom_context_num=0,
        model_type="protein_mpnn",
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
    其中 randn 用 seed 固定（每域一个解码顺序，批内 8 个 pH 共享）。
    """
    L = feature_dict["X"].shape[1]
    fd = {k: v.to(device) if torch.is_tensor(v) else v for k, v in feature_dict.items()}
    rng = np.random.RandomState(seed)
    randn = torch.from_numpy(rng.randn(L).astype(np.float32)).to(device)
    return {
        "X": fd["X"], "S": fd["S"], "mask": fd["mask"].float(),
        "chain_mask": fd["chain_mask"].float(),
        "R_idx": fd["R_idx"], "chain_labels": fd["chain_labels"],
        "randn": randn,
    }


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

    logln(f"=== ConfuMPNN Phase 2 条件微调启动 ===")
    logln(f"device={device}  epochs={args.epochs}  lr={args.lr}  "
          f"λ_c={args.lambda_c}  λ_kl={args.lambda_kl}  λ_keep={args.lambda_keep}  "
          f"perturb_prob={args.perturb_prob}  perturb_scale={args.perturb_scale}  "
          f"charge_temp={args.charge_temp}")

    # ---- backbone + 条件编码器 ----
    backbone = load_backbone(args.weights, device)
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

    # 预解析每个结构域 → 缓存 feature 张量 + encode + 无条件 logits
    # （backbone 冻结 → encode 与无条件输出每域只需算一次，全 epoch 复用）
    # prody parsePDB 按文件后缀判断格式：无 .pdb 后缀会被当 mmCIF 解析。
    # CATH 文件无扩展名 → 用 .pdb 后缀的符号链接目录（data/ 下，git 不跟踪）。
    dom_cache_dir = Path(args.dompdb).parent / (Path(args.dompdb).name + "_pdb")
    dom_cache_dir.mkdir(exist_ok=True)
    abs_dompdb = os.path.abspath(args.dompdb)

    domains = []
    for i, did in enumerate(domain_ids[:n_dom]):
        link_path = dom_cache_dir / f"{did}.pdb"
        if not link_path.exists():
            os.symlink(os.path.join(abs_dompdb, str(did)), link_path)
        pdb_path = str(link_path)
        protein_dict, *_ = parse_PDB(pdb_path, device="cpu", parse_all_atoms=False)
        L = protein_dict["X"].shape[0]
        # 单链 CATH 域：全部残基设计 → chain_mask = 全 1
        protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
        feature_dict = featurize(
            protein_dict, use_atom_context=False, number_of_ligand_atoms=0,
            model_type="protein_mpnn",
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
        # 该域 8 个 (pH, charge) 条件
        idx0 = i * n_pH
        dom["pH"] = pH_arr[idx0:idx0 + n_pH]
        dom["charge_label"] = charge_arr[idx0:idx0 + n_pH]
        domains.append(dom)
        if (i + 1) % 200 == 0:
            logln(f"  预解析+encode {i+1}/{n_dom}")

    # 记录每域梯度归零的分界（backbone 参数全 requires_grad=False，
    # 只传 enc.parameters() 给 optimizer，天然只更新编码器）
    total_cached = sum(d["h_E"].numel() for d in domains) * 4 / 1e9
    logln(f"预解析完成，缓存 encode 特征 ~{total_cached:.2f}GB")

    # ---- 训练 ----
    domain_idx = list(range(n_dom))
    n_steps_total = args.epochs * n_dom
    step = 0
    t_start = time.time()
    logln(f"开始训练：{args.epochs} epochs × {n_dom} 域/epoch = {n_steps_total} steps")

    for epoch in range(1, args.epochs + 1):
        random.shuffle(domain_idx)
        epoch_loss, epoch_ce, epoch_cd, epoch_kl, epoch_keep = [], [], [], [], []
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
                offset = torch.where(
                    mask_p,
                    torch.randint(1, int(args.perturb_scale) + 1, (B,), device=device).float()
                    * torch.where(torch.rand(B, device=device) < 0.5, 1.0, -1.0),
                    torch.zeros(B, device=device),
                )
                charge_b = charge_b + offset
            cond_b = torch.stack([
                make_condition_vector(p, c) for p, c in zip(pH_b.tolist(), charge_b.tolist())
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
            cd = torch.zeros(B, device=device)
            for i in range(B):
                cd[i] = charge_deviation_loss(
                    logits[i:i+1], pH=pH_b[i], target_charge=charge_b[i],
                    mask=ce_mask[i:i+1], temperature=args.charge_temp,
                )
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

            total = ce + args.lambda_c * cd + args.lambda_kl * kl + args.lambda_keep * keep

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
        msg = (f"epoch {epoch}/{args.epochs}  total={avg(epoch_loss):.4f}  "
               f"ce={avg(epoch_ce):.4f}  charge={avg(epoch_cd):.4f}  "
               f"kl={avg(epoch_kl):.4f}  keep={avg(epoch_keep):.4f}  "
               f"elapsed={((time.time()-t_start)/60):.1f}min")
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
            # 追溯字段（第十四轮修正参数）
            "perturb_prob": args.perturb_prob,
            "lambda_keep": args.lambda_keep,
            "charge_temp": args.charge_temp,
        }, ckpt_path)
        # 保留最新一份 alias，方便推理时加载
        torch.save(enc.state_dict(), out_dir / "condition_encoder_last.pt")

    logln(f"训练完成。总耗时 {((time.time()-t_start)/60):.1f}min。checkpoint 在 {out_dir}/")
    log.close()


if __name__ == "__main__":
    main()

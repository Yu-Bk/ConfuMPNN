"""per-epoch 前向验证损失回放（train-loss vs val-loss 曲线数据）。

对已训版本（v12.2 蛋白 / v12.3 蛋白 / v14 配体）的每个 epoch checkpoint，
在**全量验证集**上做一次 **no-grad 确定性前向**：
  - 绝不生成序列、绝不采样、无随机扰动、无 placeholder；
  - 验证集每域取其 npz 里那 8 个 (pH, charge) 臂（= native 序列在该 pH 下自身电荷，
    PROPKA 滴定，"自洽/原生锚"臂）；
  - 每 arm：CE(→native)、charge_deviation(|期望电荷−target|，logits + 该版 charge_temp)、
    native 回收（logits argmax identity，确定性非采样）；
  - 再用与训练**严格同口径**的 self-arm 分支把其它训练损失项与总损失拼出来
    （训练中只在扰动/占位臂施加的项不计入；self 臂也施加的项全计入）。

口径保证：复用 code/train_finetune.py 的 decoder_forward / build_domain / kl_anchor_loss /
load_backbone（模块内 import train_finetune as TF），以及 src 里同一批损失函数；
逐版本训练配置（flag/λ/temperature/mode/backbone/特征参数）见 code/tests/val_replay_configs.md，
脚本内置 VAL_TAGS 默认值与该文档一一对应；命令行可覆盖。

用法（项目根）：
  python code/tests/val_loss_curve.py --tag v12_2 \
      --ckpt_dir output/finetune_v12_2 --epoch_list 1,10,30 \
      --labels data/cath/labels_holdout_train.npz --dompdb data/cath/S40/dompdb_pdb \
      [--supp_labels data/cath/labels_v12_3_valsupp.npz --supp_dompdb data/cath/S40/dompdb_valsupp] \
      --device cuda:6 --out output/val_loss_curve_v12_2.json

  # 或 start/end/step：
  python code/tests/val_loss_curve.py --tag v12_2 \
      --ckpt_dir output/finetune_v12_2 --start_epoch 1 --end_epoch 30 --epoch_step 2 \
      --labels ... --dompdb ... --device cuda:6 --out ...

  # 冒烟（少域/少 epoch，可 CPU）：
  python code/tests/val_loss_curve.py --tag v14_ligand \
      --ckpt_dir output/finetune_ligand_v14_rna --epoch_list 1,50 \
      --labels data/ligand_train/labels_v14_valset_805.npz \
      --dompdb data/ligand_train/v14_valset_pdb --n_dom 3 --device cpu --out smoke.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---- 路径注入：与 train_finetune.py 一致（LigandMPNN 目录优先）----
_PROJECT_DIR = next(p for p in Path(__file__).resolve().parents
                    if (p / "code").is_dir() and (p / "LigandMPNN").is_dir())
_CODE_DIR = _PROJECT_DIR / "code"
for _p in [str(_CODE_DIR), str(_PROJECT_DIR / "LigandMPNN")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_utils import featurize, parse_PDB  # noqa: E402
import train_finetune as TF  # noqa: E402
from run_guided import load_condition_encoder  # noqa: E402

from src.condition_embedding import make_condition_vector  # noqa: E402
from src.conditioned_sampler import inject_prompt  # noqa: E402
from src.losses import (  # noqa: E402
    charge_deviation_loss, cross_entropy_loss, sequence_keep_loss,
)
from src.v10_losses import ph_aware_structure_penalty  # noqa: E402
from src.v12_losses import (  # noqa: E402
    surface_composition_loss, surface_gravy_loss, surface_charge_target_loss,
    pocket_count_loss, KD, D_IDX, E_IDX, K_IDX, R_IDX,
)
from src.structure_aware_filter import StructureAwareFilter  # noqa: E402
from src.differentiable_charge import net_charge_from_logits  # noqa: E402

# ----------------------------------------------------------------------------
# 逐版本训练配置（权威来源 = 各训练 log；self-arm 总损失装配判定见
# code/tests/val_replay_configs.md；此处默认值 = 该文档值，可被命令行覆盖）
# ----------------------------------------------------------------------------
_MOMPNN_W = (str(_PROJECT_DIR / "MoMPNN" / "mompnn_paper_checkpoints"
                   / "mompnn_temberture_tm_esm_6_4_4_b01.ckpt"))
_LIG_W = str(_PROJECT_DIR / "LigandMPNN" / "model_params" / "ligandmpnn_v_32_010_25.pt")

VAL_TAGS = {
    # v12.2 蛋白本体（MoMPNN）——log/v12_2_train_mompnn.log
    "v12_2": dict(
        mode="protein", weights=_MOMPNN_W, num_ligand_atoms=0, epochs_default=30,
        lambda_c=0.5, lambda_kl=0.05, lambda_keep=0.5, charge_temp=0.5,
        perturb_prob=0.3, perturb_scale=4.0, placeholder_prob=0.15, curriculum=False,
        decouple_perturb=True, decouple_range=12.0,
        decouple_absolute=False, decouple_abs_lo=-35.0, decouple_abs_hi=20.0,
        add_supervision=False, lambda_add=0.3, sasa_threshold=0.25,
        ph_aware_filter=True, structure_boost=1.5,
        v12_supervision=True, frac_floor=0.5, gravy_margin=0.4,
        lambda_v12=0.2, lambda_target=0.2, loss_reweight=0,
        # 该版训练代码尚无 pocket 损失（log 无 [A1] 行）→ λ_pocket=0 等效关闭
        pocket_mode="keep", pocket_cutoff=8.0,
        pocket_floor=0.7, pocket_ceil=1.3, lambda_pocket=0.0,
    ),
    # v12.3 蛋白（MoMPNN）——log/v12_3_train_mompnn.log：v12.2 + A1 keep λ_pocket=0.2
    "v12_3": dict(
        mode="protein", weights=_MOMPNN_W, num_ligand_atoms=0, epochs_default=40,
        lambda_c=0.5, lambda_kl=0.05, lambda_keep=0.5, charge_temp=0.5,
        perturb_prob=0.3, perturb_scale=4.0, placeholder_prob=0.15, curriculum=False,
        decouple_perturb=True, decouple_range=12.0,
        decouple_absolute=False, decouple_abs_lo=-35.0, decouple_abs_hi=20.0,
        add_supervision=False, lambda_add=0.3, sasa_threshold=0.25,
        ph_aware_filter=True, structure_boost=1.5,
        v12_supervision=True, frac_floor=0.5, gravy_margin=0.4,
        lambda_v12=0.2, lambda_target=0.2, loss_reweight=0,
        pocket_mode="keep", pocket_cutoff=8.0,
        pocket_floor=0.7, pocket_ceil=1.3, lambda_pocket=0.2,
    ),
    # v14 配体（LigandMPNN RNA/DNA 扩充 + A1 global）——log/v14_ligand_train_stdout.log
    "v14_ligand": dict(
        mode="ligand", weights=_LIG_W, num_ligand_atoms=25, epochs_default=50,
        lambda_c=0.5, lambda_kl=0.05, lambda_keep=0.5, charge_temp=0.5,
        perturb_prob=0.3, perturb_scale=4.0, placeholder_prob=0.15, curriculum=False,
        decouple_perturb=False, decouple_range=12.0,
        decouple_absolute=True, decouple_abs_lo=-35.0, decouple_abs_hi=20.0,
        add_supervision=False, lambda_add=0.3, sasa_threshold=0.25,
        ph_aware_filter=True, structure_boost=1.5,
        v12_supervision=True, frac_floor=0.5, gravy_margin=0.4,
        lambda_v12=0.2, lambda_target=0.2, loss_reweight=0,
        pocket_mode="global", pocket_cutoff=8.0,
        pocket_floor=0.8, pocket_ceil=1.3, lambda_pocket=0.3,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="per-epoch 前向验证损失回放")
    p.add_argument("--tag", required=True, choices=list(VAL_TAGS),
                   help="版本标识（决定默认 mode/λ/温度/pocket 等）")
    p.add_argument("--mode", choices=["protein", "ligand"], default=None,
                   help="覆盖 tag 默认 mode")
    p.add_argument("--ckpt_dir", required=True, help="含 finetune_epochNNN.pt 的目录")
    p.add_argument("--start_epoch", type=int, default=1)
    p.add_argument("--end_epoch", type=int, default=None, help="默认 = tag 的 epochs_default")
    p.add_argument("--epoch_step", type=int, default=1)
    p.add_argument("--epoch_list", default=None,
                   help="只跑指定 epoch，逗号分隔如 '1,10,30'；给则覆盖 start/end/step")
    p.add_argument("--labels", required=True, help="base 验证 npz（domain_ids/pH/charge）")
    p.add_argument("--dompdb", required=True, help="base 域 PDB 目录（含 {did}.pdb）")
    p.add_argument("--supp_labels", default=None, help="补充验证 npz（可选）")
    p.add_argument("--supp_dompdb", default=None, help="补充域 PDB 目录（可选）")
    p.add_argument("--weights", default=None, help="backbone 权重；默认取 tag")
    p.add_argument("--num_ligand_atoms", type=int, default=None, help="默认取 tag（v14=25）")
    p.add_argument("--n_dom", type=int, default=0, help="每子集最多取前 N 域（0=全部；冒烟用）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True, help="输出 JSON")
    return p.parse_args()


def resolve_epochs(args, ckpt_dir):
    """返回要处理的 epoch 列表；自动跳过缺失 checkpoint。"""
    if args.epoch_list:
        eps = []
        for s in args.epoch_list.split(","):
            s = s.strip()
            if s:
                eps.append(int(s))
    else:
        end = args.end_epoch if args.end_epoch else args.epochs_default
        eps = list(range(args.start_epoch, end + 1, args.epoch_step))
        if eps and eps[-1] != end:
            eps.append(end)          # 保证末 epoch 一定在列
    exist = [e for e in eps if (ckpt_dir / f"finetune_epoch{e:03d}.pt").is_file()]
    miss = [e for e in eps if e not in exist]
    if miss:
        print(f"  !! 缺 checkpoint，跳过: {miss}", flush=True)
    return exist


# ----------------------------------------------------------------------------
# 数据读取
# ----------------------------------------------------------------------------
def load_npz_labels(path):
    d = np.load(path, allow_pickle=True)
    ids = [str(x) for x in d["domain_ids"]]
    n_dom = len(ids)
    n_pH = d["pH"].shape[0] // n_dom
    return {
        "domain_ids": ids,
        "pH": np.asarray(d["pH"], dtype=np.float64).reshape(n_dom, n_pH),
        "charge": np.asarray(d["charge"], dtype=np.float32).reshape(n_dom, n_pH),
    }


def find_pdb(did, dompdb, dompdb2=None):
    """在 dompdb / supp 目录找 {did}.pdb（did 本身带 .pdb 也可）。"""
    fname = did if did.endswith(".pdb") else f"{did}.pdb"
    for d in (dompdb, dompdb2):
        if not d:
            continue
        cand = Path(d) / fname
        if cand.is_file():
            return cand
    return None


def pick_domains(base, supp, n_dom_cap, seed):
    """返回 [(setname, idx)]。base 在前 supp 在后；n_dom 上限作用于每子集。"""
    picked = []
    for setname, src in (("base", base), ("supp", supp)):
        if src is None:
            continue
        n = len(src["domain_ids"])
        if n_dom_cap > 0:
            n = min(n, n_dom_cap)
        picked += [(setname, i) for i in range(n)]
    return picked


# ----------------------------------------------------------------------------
# 域预解析（与 train_finetune main() 预解析同口径，只做一次、各 epoch 复用）
# ----------------------------------------------------------------------------
@torch.no_grad()
def prepare_domain(did, pdb_path, idx, backbone, device, A, ligand, num_lig, log):
    """parse+featurize+build_domain+encode+uncond logits+SASA/pocket（epoch 无关部分）。"""
    try:
        protein_dict, *_ = parse_PDB(str(pdb_path), device="cpu", parse_all_atoms=False)
    except Exception as e:
        log(f"  !! {did} parse_PDB 失败: {e}")
        return None
    L = protein_dict["X"].shape[0]
    protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
    try:
        feature_dict = featurize(
            protein_dict,
            use_atom_context=ligand,
            number_of_ligand_atoms=(num_lig if ligand else 0),
            model_type=("ligand_mpnn" if ligand else "protein_mpnn"),
        )
    except Exception as e:
        log(f"  !! {did} featurize 失败: {e}")
        return None
    dom = TF.build_domain(feature_dict, device, seed=A.seed + idx)
    # 冻结 backbone：一次性 encode
    h_V, h_E, E_idx = backbone.encode(dom)
    dom["h_V"], dom["h_E"], dom["E_idx"] = h_V, h_E, E_idx
    with torch.no_grad():
        logits_uncond = TF.decoder_forward(backbone, h_V, h_E, E_idx, dom, 1, device)
    dom["logits_uncond"] = logits_uncond
    anchor = logits_uncond[0].argmax(-1)                      # [L]
    anchor = torch.where(dom["S"][0] < 20, anchor, torch.zeros_like(anchor))
    dom["seq_anchor"] = anchor
    valid = (dom["S"] < 20).float()
    dom["ce_mask"] = dom["mask"] * dom["chain_mask"] * valid
    dom["native_int"] = dom["S"][0].long().cpu().numpy()       # [L]
    dom["domain_id"] = did
    dom["L"] = L
    # v12 系监督需要 SASA（训练侧同款；失败 → 该域跳过 v12 项）
    use_sasa = bool(A.v12_supervision or A.add_supervision)
    if use_sasa:
        try:
            from src.sasa import fractional_sasa
            sasa_info = fractional_sasa(str(pdb_path),
                                        surface_threshold=A.sasa_threshold,
                                        align_to_full=False)
            sasa_frac = sasa_info["frac_sasa"]
            sasa_resids = sasa_info["residues"]
            dom_resids = np.asarray(dom["R_idx"].cpu().numpy()).reshape(-1)
            sasa_map = {int(r): i for i, r in enumerate(sasa_resids)}
            aligned = np.zeros(L, dtype=np.float64)
            for pos in range(L):
                rid = int(dom_resids[pos])
                if rid in sasa_map:
                    aligned[pos] = sasa_frac[sasa_map[rid]]
            dom["frac_sasa"] = aligned
            # pocket 分区（三块互斥）。protein 无 Y → pocket=0。
            if A.pocket_mode in ("keep", "global"):
                Y = dom.get("Y")
                CA = dom["X"][0, :, 1].cpu().numpy()
                if Y is not None and Y.numel() > 0:
                    Yc = Y.reshape(-1, 3).cpu().numpy()
                    dmin = np.linalg.norm(CA[:, None, :] - Yc[None, :, :], axis=-1).min(axis=1)
                    pocket = (dmin < A.pocket_cutoff).astype(np.float32)
                else:
                    pocket = np.zeros(L, dtype=np.float32)
                surf = aligned >= A.sasa_threshold
                core = ((~surf) & (pocket == 0)).astype(np.float32)
                dom["pocket_mask"] = pocket
                dom["core_mask"] = core
                dom["charge_surf_mask"] = np.clip(surf.astype(np.float32) + pocket, 0.0, 1.0)
        except Exception as e:
            log(f"  ⚠️ {did} SASA/pocket 分区失败: {e}（跳过 v12 监督项）")
            dom["frac_sasa"] = None
    # pH 自适应结构过滤器（ph_aware_filter 用；coords 固定 → 每域实例化一次缓存）
    if A.ph_aware_filter:
        try:
            coords = dom["X"][0, :, 1].cpu().numpy()   # [L,3] Cα
            dom["filter"] = StructureAwareFilter(coords)
        except Exception as e:
            log(f"  ⚠️ {did} StructureAwareFilter 构造失败: {e}")
            dom["filter"] = None
    return dom


def parse_prepare_all(A, backbone, device):
    """读 labels、找 PDB、prepare_domain；返回 [(did, dom)] 列表。"""
    base = load_npz_labels(A.labels)
    supp = load_npz_labels(A.supp_labels) if A.supp_labels else None
    picks = pick_domains(base, supp, A.n_dom, A.seed)

    def _log(m):
        print(m, flush=True)

    out = []
    for setname, i in picks:
        src = base if setname == "base" else supp
        did = src["domain_ids"][i]
        p = find_pdb(did, A.dompdb, A.supp_dompdb)
        if p is None:
            _log(f"  !! {did} 找不到 PDB，跳过")
            continue
        dom = prepare_domain(did, p, len(out), backbone, device, A,
                             A.mode_ligand, A.num_ligand_atoms, _log)
        if dom is None:
            continue
        # 该域 n_arm 个 (pH, charge) 自洽臂
        dom["pH"] = torch.from_numpy(src["pH"][i].astype(np.float32)).to(device)
        dom["charge_label"] = torch.from_numpy(src["charge"][i]).to(device)
        dom["n_arm"] = int(src["pH"].shape[1])
        out.append((did, dom))
    print(f"预解析成功 {len(out)}/{len(picks)} 域", flush=True)
    return out


# ----------------------------------------------------------------------------
# self-arm 前向 + 损失装配（严格镜像 train_finetune.py self 分支，行号依据见
# code/tests/val_replay_configs.md）
# ----------------------------------------------------------------------------
@torch.no_grad()
def replay_domain(dom, backbone, enc, device, A):
    """对单个域做一次确定性 no-grad 前向，返回各项标量。全自洽臂（B=n_arm）。"""
    B = dom["n_arm"]
    pH_b = dom["pH"]                  # [B] float32
    charge_b = dom["charge_label"]    # [B] float32
    cond_b = torch.stack([
        make_condition_vector(float(p), net_charge=float(c))
        for (p, c) in zip(pH_b.tolist(), charge_b.tolist())
    ]).to(device)
    prompt = enc(cond_b)                                  # [B,4,128]
    h_V = dom["h_V"].repeat(B, 1, 1)
    h_V = inject_prompt(h_V, prompt)
    logits = TF.decoder_forward(backbone, h_V, dom["h_E"], dom["E_idx"], dom, B, device)

    S_true = dom["S"].long().repeat(B, 1)
    ce_mask = dom["ce_mask"].repeat(B, 1)
    ce = cross_entropy_loss(logits, S_true, ce_mask).item()

    # ---- 电荷偏差（逐 arm，temperature=该版 charge_temp；对应训练 696-714 的 self 分支）----
    cd_arms = []
    for i in range(B):
        cd_i = charge_deviation_loss(
            logits[i:i+1], pH=float(pH_b[i]), target_charge=float(charge_b[i]),
            mask=ce_mask[i:i+1], temperature=A.charge_temp)
        cd_arms.append(cd_i.item())
    cd = float(np.mean(cd_arms))

    # ---- native 回收（logits argmax identity，确定性非采样）----
    S_hat = logits.argmax(-1)                              # [B,L]
    valid = (dom["S"][0] < 20).float().to(device)          # [L]
    rec_i = ((S_hat == dom["S"]) * valid.unsqueeze(0)).sum(-1) / valid.sum().clamp(min=1.0)
    rec = float(rec_i.mean())

    # ---- KL 锚（条件化 ‖ 无条件，对应 717-718）----
    ref = dom["logits_uncond"].repeat(B, 1, 1)
    kl = TF.kl_anchor_loss(logits, ref, ce_mask).item() if A.lambda_kl > 0 else 0.0

    # ---- SeqKeep（仅自洽样本；回放全自洽 → 全部施加，对应 722-729）----
    keep_arms = []
    if A.lambda_keep > 0:
        anchor = dom["seq_anchor"].unsqueeze(0)            # [1,L]
        for i in range(B):
            keep_arms.append(sequence_keep_loss(logits[i:i+1], anchor, ce_mask[i:i+1]).item())
    keep = float(np.mean(keep_arms)) if keep_arms else 0.0

    frac = dom.get("frac_sasa")
    has_sasa = frac is not None

    # ---- v12：组成 + GRAVY（回放无占位 → B 全算；对应 778-799）----
    v12_comp = v12_gravy = 0.0
    if A.v12_supervision and has_sasa:
        surf = frac >= A.sasa_threshold
        nat_int = dom["native_int"]
        sel = nat_int[surf]
        sel = sel[sel < 20]
        native_grav = float(KD.cpu().numpy()[sel].mean()) if len(sel) > 0 else 0.0
        comp_arms, gravy_arms = [], []
        for i in range(B):
            comp_arms.append(surface_composition_loss(
                logits[i:i+1], frac, nat_int, frac_floor=A.frac_floor,
                surface_threshold=A.sasa_threshold).item())
            gravy_arms.append(surface_gravy_loss(
                logits[i:i+1], frac, native_grav, margin=A.gravy_margin,
                surface_threshold=A.sasa_threshold).item())
        v12_comp = float(np.mean(comp_arms))
        v12_gravy = float(np.mean(gravy_arms))

    # ---- v12.2 表面电荷目标（锚净电荷 = target − 核心 native 电荷；对应 804-830）----
    v12_ct = 0.0
    if A.lambda_target > 0 and has_sasa:
        if A.pocket_mode in ("keep", "global") and dom.get("core_mask") is not None:
            core_mask = dom["core_mask"]
            charge_mask = dom["charge_surf_mask"]
        else:
            surf = frac >= A.sasa_threshold
            core_mask = (~surf).astype(np.float32)
            charge_mask = None
        nat_int = dom["native_int"]
        nat_onehot = F.one_hot(torch.as_tensor(nat_int).clamp(0, 19).long(),
                               num_classes=20).float().to(device)
        ct_arms = []
        for i in range(B):
            q_core = net_charge_from_logits(
                nat_onehot.unsqueeze(0), pH=float(pH_b[i]),
                mask=torch.as_tensor(core_mask, device=device).unsqueeze(0),
                include_termini=False)
            target_surf = float(charge_b[i]) - float(q_core)
            ct_arms.append(surface_charge_target_loss(
                logits[i:i+1], pH=float(pH_b[i]), target_surface_charge=target_surf,
                frac_sasa=frac, surface_threshold=A.sasa_threshold,
                temperature=A.charge_temp, extra_mask=charge_mask).item())
        v12_ct = float(np.mean(ct_arms))

    # ---- v10 C：pH 自适应结构惩罚（self 臂 boost=1.0；对应 755-771）----
    struct = 0.0
    if A.ph_aware_filter and dom.get("filter") is not None:
        filt = dom["filter"]
        seq_int_cur = dom["native_int"]
        sp_arms = []
        for i in range(B):
            sp_i, _ = ph_aware_structure_penalty(
                logits[i:i+1], filt, seq_int_cur,
                pH=float(pH_b[i]), mask=ce_mask[i:i+1], scale_boost=1.0)
            sp_arms.append(sp_i.item())
        struct = float(np.mean(sp_arms))

    # ---- A1 双向计数（蛋白无配体 → pocket 区 0 → 该项≈0；配体 v14 生效；对应 838-860）----
    pocket = 0.0
    if (A.pocket_mode in ("keep", "global") and A.lambda_pocket > 0
            and dom.get("pocket_mask") is not None and has_sasa):
        if A.pocket_mode == "global":
            region = dom["charge_surf_mask"].astype(bool)
        else:
            region = dom["pocket_mask"].astype(bool)
        nat_int = dom["native_int"]
        nat_neg = int(((nat_int == D_IDX) | (nat_int == E_IDX))[region].sum())
        nat_pos = int(((nat_int == K_IDX) | (nat_int == R_IDX))[region].sum())
        region_f = region.astype(np.float32)
        pk_arms = []
        for i in range(B):
            pk_arms.append(pocket_count_loss(
                logits[i:i+1], region_f, (nat_neg, nat_pos),
                floor=A.pocket_floor, ceil=A.pocket_ceil,
                normalize=(A.pocket_mode == "global")).item())
        pocket = float(np.mean(pk_arms))

    # ---- 总损失装配（train_finetune 行 ~862-872）----
    # 注：v10 B add_supervision 只对扰动臂施加 + 三版均未开 → self 回放不计（见 val_replay_configs.md §0）
    total = ce + A.lambda_c * cd + A.lambda_kl * kl + A.lambda_keep * keep
    if A.ph_aware_filter:
        total = total + 0.05 * struct
    if A.v12_supervision:
        total = total + A.lambda_v12 * (v12_comp + v12_gravy)
        if A.lambda_target > 0:
            total = total + A.lambda_target * v12_ct
    if A.pocket_mode in ("keep", "global") and A.lambda_pocket > 0:
        total = total + A.lambda_pocket * pocket

    return {"ce": ce, "cd": cd, "rec": rec, "kl": kl, "keep": keep,
            "v12_comp": v12_comp, "v12_gravy": v12_gravy, "v12_ct": v12_ct,
            "struct": struct, "pocket": pocket, "total": float(total)}


def main():
    args = parse_args()
    cfg0 = VAL_TAGS[args.tag]
    # 合并：命令行已给的值优先，未给(None/缺省)则用 tag 默认
    for k, v in cfg0.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    args.mode_ligand = (args.mode == "ligand")
    args.ckpt_dir = Path(args.ckpt_dir)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"tag={args.tag} mode={args.mode}  device={device}", flush=True)

    epochs = resolve_epochs(args, args.ckpt_dir)
    if not epochs:
        raise SystemExit(f"{args.ckpt_dir} 下无可用 epoch checkpoint，退出")
    print(f"epochs to replay: {epochs}", flush=True)

    backbone = TF.load_backbone(args.weights, device, ligand=args.mode_ligand)
    backbone.eval()
    print(f"backbone loaded: {Path(args.weights).name}  (mode={args.mode})", flush=True)

    t0 = time.time()
    domains = parse_prepare_all(args, backbone, device)
    if not domains:
        raise SystemExit("无可回放域，退出")
    print(f"预解析 {len(domains)} 域 耗时 {(time.time()-t0)/60:.1f}min", flush=True)

    meta = {"tag": args.tag, "mode": args.mode, "labels": args.labels,
            "supp_labels": args.supp_labels, "dompdb": args.dompdb,
            "seed": args.seed, "epochs": epochs,
            "lambda_c": args.lambda_c, "lambda_kl": args.lambda_kl,
            "lambda_keep": args.lambda_keep, "charge_temp": args.charge_temp,
            "v12_supervision": args.v12_supervision, "frac_floor": args.frac_floor,
            "gravy_margin": args.gravy_margin, "lambda_v12": args.lambda_v12,
            "lambda_target": args.lambda_target, "sasa_threshold": args.sasa_threshold,
            "ph_aware_filter": args.ph_aware_filter,
            "pocket_mode": args.pocket_mode, "lambda_pocket": args.lambda_pocket,
            "pocket_floor": args.pocket_floor, "pocket_ceil": args.pocket_ceil,
            "n_domains": len(domains),
            "note": "全自洽臂 no-grad 前向回放；per-epoch 配置依据见 code/tests/val_replay_configs.md"}
    results = {"meta": meta, "epochs": {}}

    for ep in epochs:
        ckpt = args.ckpt_dir / f"finetune_epoch{ep:03d}.pt"
        enc = load_condition_encoder(str(ckpt), device)
        acc = {}
        for did, dom in domains:
            r = replay_domain(dom, backbone, enc, device, args)
            for k, v in r.items():
                acc.setdefault(k, []).append(v)
        ep_stat = {k: float(np.mean(v)) for k, v in acc.items()}
        ep_stat["n_dom"] = len(domains)
        ep_stat["n_arm"] = int(sum(d["n_arm"] for _, d in domains))
        results["epochs"][str(ep)] = ep_stat
        print(f"epoch {ep:3d}: ce={ep_stat['ce']:.4f} cd={ep_stat['cd']:.4f} "
              f"rec={ep_stat['rec']:.4f} total={ep_stat['total']:.4f} "
              f"(n_dom={ep_stat['n_dom']})", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"已写 {args.out}", flush=True)


if __name__ == "__main__":
    main()

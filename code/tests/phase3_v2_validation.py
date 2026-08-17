"""Phase 3 v2 复验采样：目标 1（天然骨架+位点固定）+ 目标 2（从零设 pI + 占位符）。

对应 `session/2026-08-17_validation_plan_v2.md` §5（对齐用户两个真实目标）。
使用**第十五轮重训编码器**（finetune_v2，perturb_scale=8 + placeholder_prob=0.15）。

设计（每 PDB，n=20，seed=111+k，固定 seed 协议防挑 seed）：
    t1_cond : pH7.4 target=round(native)  +固定 4 个疏水核心位点  → 目标1 形态，判 H2/S4/S1*/H1
    t1_base : pH7.4 target=round(native)   不固定                  → S1* 参照 + 对照
    t2_pos  : pH7.4 target=round(native)+5 不固定                  → 目标2 从零设 pI（正电臂），判 H2
    t2_neg  : pH7.4 target=round(native)-5 不固定                  → 目标2 从零设 pI（负电臂），判 H2
    t2_ph   : pH7.4 target=None（占位符）   不固定                  → 目标2 占位符语义，判 S3

占位符语义：make_condition_vector(pH, net_charge=None) → has_charge=0、charge=0
（与训练注入的「flag=0+值0」占位样本一致）。

输出目录 output/finetune_v2_validate/：
    {pdb}/{arm}/seqs.fa           每臂序列（header 含 seed/pH/target/charge）
    {pdb}/native.fa               native 参照
    {pdb}/charge_stats.json       每臂电荷命中统计（H2 初判）
    {pdb}/diversity.json          防坍塌：pairwise identity + 位置熵

用法（code/ 下）：
    PYTHONPATH=. python tests/phase3_v2_validation.py \
        --cond_encoder output/finetune_v2/finetune_epoch030.pt \
        --out_dir output/finetune_v2_validate [--n 20] [--dry-run 1]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml  # noqa: E402

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_CODE_DIR.parent / "LigandMPNN"))

from data_utils import featurize, parse_PDB  # noqa: E402
from src.condition_embedding import make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from run_guided import _DEFAULT_WEIGHTS, load_condition_encoder, load_model, seq_to_string  # noqa: E402

# 验证 PDB 集合（与 E1b / Phase 3 一致，全部经泄漏检查不在训练集）
PDBS = {
    "1BC8": "input/1BC8_chainC.pdb",
    "1CRN": "input/1CRN.pdb",
    "1UBQ": "input/1UBQ.pdb",
    "2LZM": "input/2LZM.pdb",
    # 第十八轮新增：正电验证蛋白（native charge +8.0，未在训练集，测通用模型泛化）
    "1b24A01": "input/1b24A01.pdb",
    "1a87A02": "input/1a87A02.pdb",
}
AA1 = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC = set("ILVFMYW")  # 疏水核心位点候选（保守位点代理）

# 电荷维度训练均值（condition_defaults.yaml normalization.mean[2]）
# 第十七轮：占位符统一用"均值占位"（has_charge=1 + 值=训练均值），符合目标 2"非 0 占位符"
with open(_CODE_DIR / "configs" / "condition_defaults.yaml") as _f:
    _CFG = yaml.safe_load(_f)["condition_defaults"]
CHARGE_MEAN = float(_CFG["normalization"]["mean"][2])


def pick_fixed_sites(seq, residue_names, k=4):
    """选 k 个均匀分布的疏水核心位点作为保守固定位点。

    规则：按序列 1-based 位置等分 k 段，每段取最近的疏水残基（I/L/V/F/M/Y/W）。
    若某段无疏水残基，退回段中心残基（保证不破坏 S4 验证的最小固定）。
    返回 residue_names 的子集（如 ['C5','C15',...]）。
    """
    hphob_idx = [i for i, aa in enumerate(seq) if aa in HYDROPHOBIC]
    L = len(seq)
    chosen = []
    for seg in range(k):
        lo, hi = int(seg * L / k), int((seg + 1) * L / k)  # [lo, hi) 半开
        best = None
        for i in hphob_idx:
            if lo <= i < hi:
                best = i
                break
        if best is None:  # 段内无疏水残基 → 段中心
            best = (lo + hi) // 2
        chosen.append(residue_names[best])
    return sorted(chosen, key=lambda s: int("".join(c for c in s if c.isdigit())))


def build_feature_dict(pdb_path, device, fixed_ids=None):
    protein_dict, _, _, icodes, _ = parse_PDB(pdb_path)
    protein_dict["chain_mask"] = torch.ones(
        protein_dict["X"].shape[0], dtype=torch.int32
    )
    if fixed_ids:
        R_idx = list(protein_dict["R_idx"].cpu().numpy())
        chain_letters = list(protein_dict["chain_letters"])
        encoded = [
            str(chain_letters[i]) + str(R_idx[i]) + icodes[i]
            for i in range(len(R_idx))
        ]
        fixed_set = set(fixed_ids)
        for i, name in enumerate(encoded):
            if name in fixed_set:
                protein_dict["chain_mask"][i] = 0
    feature_dict = featurize(
        protein_dict, cutoff_for_score=8.0,
        use_atom_context=False, number_of_ligand_atoms=0,
        model_type="protein_mpnn",
    )
    feature_dict["batch_size"] = 1
    feature_dict["temperature"] = 0.3
    L = feature_dict["X"].shape[1]
    feature_dict["bias"] = torch.zeros(1, L, 21)
    return feature_dict


def sample_one(model, enc, fd, cond_vec, device, seed):
    """固定 seed 采一条条件序列。"""
    torch.manual_seed(seed)
    L = fd["X"].shape[1]
    fd["randn"] = torch.randn(1, L)
    out = conditioned_sample(model, enc, fd, cond_vec, device=device)
    return seq_to_string(out["S"][0].cpu().numpy())


def write_fasta(path, header, seq):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f">{header}\n{seq}\n")


def compute_diversity(seqs):
    """防坍塌：pairwise identity + 每位置熵。seqs: list[str]（同长度）。"""
    n, L = len(seqs), len(seqs[0])
    # pairwise identity（两两比较，统计 n² 中同位置同 AA 比例的平均）
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = seqs[i], seqs[j]
            total += sum(a == b for a, b in zip(si, sj)) / L
            pairs += 1
    pairwise_id = total / pairs if pairs else float("nan")
    # 每位置熵（20 种 AA 分布，香农熵，单位 bit）
    entropy_sum = 0.0
    for pos in range(L):
        counts = [0] * 20
        for s in seqs:
            counts[AA1.index(s[pos])] += 1
        p = np.array(counts) / n
        p = p[p > 0]
        entropy_sum -= float(np.sum(p * np.log2(p)))
    return {"n": n, "L": L, "pairwise_identity": round(pairwise_id, 4),
            "mean_position_entropy": round(entropy_sum / L, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_dir", default=str(_CODE_DIR))
    ap.add_argument("--cond_encoder", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed_base", type=int, default=111)
    ap.add_argument("--pH", type=float, default=7.4)
    ap.add_argument("--dry_run", type=int, default=0,
                    help=">0 时每臂只采该数量样本（冒烟用）")
    ap.add_argument("--device", default="cuda:1",
                    help="计算设备（默认 cuda:1，GPU 0 被占）")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_root = Path(args.out_dir)
    n = args.n if args.dry_run <= 0 else args.dry_run

    weights = args.weights if args.weights else str(_DEFAULT_WEIGHTS)
    print(f"[load] model={Path(weights).name} device={device}", flush=True)
    model = load_model(weights, device, model_type="protein_mpnn")
    enc = load_condition_encoder(args.cond_encoder, device)

    for pdb, rel in PDBS.items():
        pdb_path = os.path.join(args.pdb_dir, rel)
        fd0 = build_feature_dict(pdb_path, device)   # 不含固定位点
        L = fd0["X"].shape[1]
        native = seq_to_string(fd0["S"][0].cpu().numpy())
        native_charge = net_charge(native, args.pH)
        t_nat = round(native_charge)

        # 选固定位点（仅 t1_cond 用）：从 PDB 解析真实残基名（链字母+残基号+icode）
        protein_dict, _, _, icodes, _ = parse_PDB(pdb_path)
        rr = list(protein_dict["R_idx"].cpu().numpy())
        cl = list(protein_dict["chain_letters"])
        resnames = [str(cl[i]) + str(rr[i]) + icodes[i] for i in range(len(rr))]
        fixed_ids = pick_fixed_sites(native, resnames, k=4)

        scenarios = {
            "t1_cond": (args.pH, float(t_nat), fixed_ids),
            "t1_base": (args.pH, float(t_nat), None),
            "t2_pos":  (args.pH, float(t_nat + 3), None),   # 温和正电（用户建议 target 不设极端）
            "t2_pos_extreme": (args.pH, float(t_nat + 8), None),  # 极端正电（第十八轮核心：验证分层采样+逆加权后高正电外推）
            "t2_neg":  (args.pH, float(t_nat - 5), None),
            "t2_ph":   (args.pH, float(CHARGE_MEAN), None),  # 均值占位（非 0 占位符：has_charge=1+值=训练均值）
        }
        print(f"\n=== {pdb} L={L} native@{args.pH}={native_charge:+.2f} "
              f"t_nat={t_nat} 固定位点={fixed_ids} ===", flush=True)

        # native 参照
        write_fasta(out_root / pdb / "native.fa",
                    f"native charge={native_charge:+.2f}", native)

        stats = {}
        for arm, (pH, tgt, fids) in scenarios.items():
            fd = build_feature_dict(pdb_path, device, fixed_ids=fids)
            fa = out_root / pdb / arm / "seqs.fa"
            fa.parent.mkdir(parents=True, exist_ok=True)
            if fa.exists():
                fa.unlink()
            charges, seqs = [], []
            for k in range(n):
                seed = args.seed_base + k
                cond_vec = make_condition_vector(pH, net_charge=tgt)
                seq = sample_one(model, enc, fd, cond_vec, device, seed)
                q = net_charge(seq, pH)
                charges.append(q)
                seqs.append(seq)
                tgt_str = ('均值占位' if arm == 't2_ph' else
                           ('占位' if tgt is None else f'{tgt:+.1f}'))
                write_fasta(
                    fa,
                    f"seed_{seed} arm={arm} pH={pH} "
                    f"target={tgt_str} charge={q:+.2f}",
                    seq,
                )
            mean_q, std_q = float(np.mean(charges)), float(np.std(charges))
            dev = abs(mean_q - (tgt if tgt is not None else native_charge))
            stats[arm] = {
                "target": tgt, "mean_charge": round(mean_q, 2),
                "std_charge": round(std_q, 2),
                "mean_abs_dev": round(dev, 2),
                "n": n,
            }
            div = compute_diversity(seqs)
            stats[arm]["diversity"] = div
            print(f"  {arm:9s} target={tgt if tgt is not None else '占位':>5} "
                  f"mean={mean_q:+6.2f}±{std_q:.2f} dev={dev:.2f} "
                  f"pairID={div['pairwise_identity']:.2f} "
                  f"Hpos={div['mean_position_entropy']:.2f}",
                  flush=True)
        with open(out_root / pdb / "charge_stats.json", "w", encoding="utf-8") as f:
            json.dump({"pdb": pdb, "native_charge": native_charge,
                       "native": native, "fixed_ids": fixed_ids,
                       "arms": stats}, f, ensure_ascii=False, indent=2)
        print(f"  → {out_root / pdb / 'charge_stats.json'}", flush=True)

    print("\n=== 采样完成 ===")


if __name__ == "__main__":
    main()

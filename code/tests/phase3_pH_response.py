"""Phase 3 条件注入 pH 响应 Go/No-Go 实验。

验证微调后模型（MoMPNN + ConditionEncoder）是否真正感知 pH/电荷条件：

1. **target 响应**：固定 pH=7.4，变 target → 平均电荷是否单调 + 线性增益（校准）
2. **pH 响应**：固定 target=0，变 pH → 生成的序列是否不同（同一 seed 下）。
   Phase 1 诚实边界 =「同一 seed 下各 pH 序列完全相同」→ 若 identity < 100% 则模型感知了 pH
3. **序列多样性 / identity**：跨条件平均序列一致性（同 seed 对应位置比较）

关键设计：**固定 seed**。同一 seed → 解码顺序相同 → 序列差异只来自条件注入，
把「条件影响」从「采样随机性」中干净分离出来。

用法（code/ 下）：
    PYTHONPATH=. python tests/phase3_pH_response.py \
        --pdb input/1BC8.pdb \
        --cond_encoder output/finetune/condition_encoder_last.pt \
        --out_dir output/phase3/1BC8
    # --targets 可自定义（默认 [native, -5, 0, 5]@pH7.4）
    # --pHs 可自定义（默认 [4, 7.4, 9]@target=0）
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_CODE_DIR.parent / "LigandMPNN"))

from data_utils import featurize, parse_PDB  # noqa: E402
from src.condition_embedding import make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from run_guided import _DEFAULT_WEIGHTS, load_condition_encoder, load_model, seq_to_string  # noqa: E402


def build_feature_dict(model, pdb_path, device):
    protein_dict, _, _, _, _ = parse_PDB(pdb_path)
    protein_dict["chain_mask"] = torch.ones(
        protein_dict["X"].shape[0], dtype=torch.int32
    )
    feature_dict = featurize(
        protein_dict, cutoff_for_score=8.0,
        use_atom_context=False, number_of_ligand_atoms=0,
        model_type="protein_mpnn",
    )
    L = feature_dict["X"].shape[1]
    feature_dict["batch_size"] = 1
    feature_dict["temperature"] = 0.3
    feature_dict["bias"] = torch.zeros(1, L, 21)
    return feature_dict, protein_dict


def sample_one(model, enc, feature_dict, cond_vec, device, seed):
    """固定 seed 采样一条（同 seed → 解码顺序相同，条件影响可分离）。"""
    torch.manual_seed(seed)
    L = feature_dict["X"].shape[1]
    feature_dict["randn"] = torch.randn(1, L)
    out = conditioned_sample(model, enc, feature_dict, cond_vec, device=device)
    return seq_to_string(out["S"][0].cpu().numpy())


def identity(a, b):
    """两条等长序列的逐位一致率。"""
    return sum(x == y for x, y in zip(a, b)) / len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--cond_encoder", required=True)
    ap.add_argument("--weights", default=None,
                    help="backbone 权重（默认 run_guided 的 MoMPNN）")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=8, help="每条件采样数")
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--targets", type=str, default="auto",
                    help="target 响应集合（逗号分隔，auto=用 native±[5,0]）")
    ap.add_argument("--pHs", type=str, default="4.0,7.4,9.0")
    ap.add_argument("--ref_pH", type=float, default=7.4, help="target 响应用的固定 pH")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = args.weights if args.weights else str(_DEFAULT_WEIGHTS)
    model = load_model(weights, device, model_type="protein_mpnn")
    enc = load_condition_encoder(args.cond_encoder, device)
    feature_dict, protein_dict = build_feature_dict(model, args.pdb, device)
    native_seq = seq_to_string(feature_dict["S"][0].cpu().numpy())
    native_charge = net_charge(native_seq, args.ref_pH)
    L = feature_dict["X"].shape[1]
    print(f"PDB={os.path.basename(args.pdb)}  L={L}  "
          f"native@{args.ref_pH} charge={native_charge:+.2f}")

    # ---- target 集合 ----
    if args.targets == "auto":
        targets = sorted({round(native_charge + 5), round(native_charge - 5),
                          0, round(native_charge)}, reverse=True)
    else:
        targets = [float(x) for x in args.targets.split(",")]
    pHs = [float(x) for x in args.pHs.split(",")]

    results = {"pdb": args.pdb, "L": L, "native_charge": native_charge,
               "ref_pH": args.ref_pH, "n": args.n, "seed": args.seed,
               "target_response": [], "pH_response": []}

    # ---- ① target 响应（固定 ref_pH）----
    print(f"\n=== ① target 响应（pH={args.ref_pH}，固定 seed={args.seed}）===")
    for t in targets:
        cond_vec = make_condition_vector(args.ref_pH, net_charge=t)
        seqs = [sample_one(model, enc, feature_dict, cond_vec, device, args.seed + k)
                for k in range(args.n)]
        charges = [net_charge(s, args.ref_pH) for s in seqs]
        res = {"target": t, "mean_charge": float(np.mean(charges)),
               "std_charge": float(np.std(charges)),
               "charges": [round(c, 2) for c in charges]}
        results["target_response"].append(res)
        print(f"  target={t:+5.1f} → mean={res['mean_charge']:+6.2f} "
              f"± {res['std_charge']:.2f}")

    # ---- ② pH 响应（固定 target=0）----
    print(f"\n=== ② pH 响应（target=0，固定 seed={args.seed}）===")
    pH_seqs = {}
    for ph in pHs:
        cond_vec = make_condition_vector(ph, net_charge=0.0)
        seqs = [sample_one(model, enc, feature_dict, cond_vec, device, args.seed + k)
                for k in range(args.n)]
        charges = [net_charge(s, ph) for s in seqs]
        pH_seqs[ph] = seqs
        res = {"pH": ph, "mean_charge": float(np.mean(charges)),
               "std_charge": float(np.std(charges)),
               "charges": [round(c, 2) for c in charges]}
        results["pH_response"].append(res)
        print(f"  pH={ph:4.1f} → mean charge@{ph}={res['mean_charge']:+6.2f} "
              f"± {res['std_charge']:.2f}")

    # ---- ③ 跨 pH identity（同 seed 对应位置）----
    if len(pHs) >= 2:
        print(f"\n=== ③ 跨 pH 序列 identity（同 seed 对应位置，越小=越感知 pH）===")
        pH_ids = {f"{ph:.1f}": ph for ph in pHs}
        cross = {}
        for a in pHs:
            for b in pHs:
                if a < b:
                    id_vals = [identity(pH_seqs[a][k], pH_seqs[b][k])
                               for k in range(args.n)]
                    key = f"pH{a:.1f}_vs_{b:.1f}"
                    cross[key] = float(np.mean(id_vals))
                    print(f"  {key}: identity = {np.mean(id_vals):.3f}")
        results["cross_pH_identity"] = cross
        # 同 pH 内对照组（应=1.0，自检 seed 复现）
        if all(pH_seqs[ph][0] == pH_seqs[ph][0] for ph in pHs):
            pass
        same = [identity(pH_seqs[pHs[0]][k], pH_seqs[pHs[0]][k]) for k in range(args.n)]
        results["same_cond_identity_selfcheck"] = float(np.mean(same))

    with open(out_dir / "phase3_pH_response.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已写 {out_dir / 'phase3_pH_response.json'}")


if __name__ == "__main__":
    main()

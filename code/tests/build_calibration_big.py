"""v12.2 扩大校准域建表：训练域批量响应诊断 → global 校准表（无泄露）。

背景（index/PROJECT_LOCAL_V12_2.md §6E）：原校准表用 17 蛋白 204 点拟合 global，
结构域太少（用户质疑）；且 per-protein 泛化验证对评估蛋白有"响应信息泄漏"。
本脚本用 v12.2 真训练域（labels_v12_2_train.npz，6,710 域，模型见过）分层抽样
100 域，每域测 native±[8,4,0,4,8] 共 5 target × n10 → 500 点拟合 global。
**训练域不在评估集 → 天然无泄露**（⚠️ 不用 labels_balanced_v7 全量——含 hold-out 1,176 域）。

用法（项目根）：
  PYTHONPATH=code /home/baokun_yu/miniconda3/envs/confumpnn/bin/python code/tests/build_calibration_big.py --device cuda:5
输出：output/charge_calibration_v12_2_big.json（global + per_protein 100 域）
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_DIR = next(p for p in Path(__file__).resolve().parents
                    if (p / "code").is_dir() and (p / "LigandMPNN").is_dir())
_CODE_DIR = _PROJECT_DIR / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_PROJECT_DIR / "LigandMPNN"))

from data_utils import featurize, parse_PDB  # noqa: E402
from src.condition_embedding import make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from run_guided import load_model, load_condition_encoder, seq_to_string  # noqa: E402

AP = argparse.ArgumentParser()
AP.add_argument("--n_dom", type=int, default=100)
AP.add_argument("--n_per", type=int, default=10)
AP.add_argument("--seed", type=int, default=42)
AP.add_argument("--temperature", type=float, default=0.3)
AP.add_argument("--device", default="cuda:5")
AP.add_argument("--labels", default=str(_PROJECT_DIR / "data/cath/labels_v12_2_train.npz"))
AP.add_argument("--dompdb", default=str(_PROJECT_DIR / "data/cath/S40/dompdb_pdb"))
AP.add_argument("--enc", default=str(_PROJECT_DIR / "output/finetune_v12_2/finetune_epoch030.pt"))
AP.add_argument("--weights", default=str(_PROJECT_DIR / "MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt"))
AP.add_argument("--out", default=str(_PROJECT_DIR / "output/charge_calibration_v12_2_big.json"))
ARGS = AP.parse_args()

OFFSETS = [-8, -4, 0, 4, 8]   # native 电荷偏移（覆盖温和+极端区）


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx if sxx > 1e-9 else float("nan")
    b = my - a * mx
    return a, b


def main():
    device = torch.device(ARGS.device)
    d = np.load(ARGS.labels, allow_pickle=True)
    n_dom = len(d["domain_ids"])
    print(f"v12.2 训练域 {n_dom} 个，分层抽样 {ARGS.n_dom}", flush=True)

    # 分层抽样：按 charge@7.4 分 10 箱每箱均取（覆盖全电荷范围）
    c7 = np.array([float(net_charge(s, 7.4)) for s in d["seqs"]])
    qs = np.quantile(c7, np.linspace(0, 1, 11))
    rng = random.Random(ARGS.seed)
    sel = []
    per_bin = ARGS.n_dom // 10   # 每箱取 n_dom/10 个（⚠️ 原 bug：取 min(n_dom//10, len) 为 n_dom//10=671 再截断 → 只取第一箱）
    for b in range(10):
        lo, hi = qs[b], qs[b + 1]
        idx = [i for i, c in enumerate(c7) if lo <= c < hi or (b == 9 and lo <= c <= hi)]
        take = min(per_bin, len(idx))
        sel.extend(rng.sample(idx, take))
    sel = sorted(sel[:ARGS.n_dom])
    print(f"抽样 {len(sel)} 域（charge@7.4 范围 [{c7[sel].min():.1f},{c7[sel].max():.1f}]）", flush=True)

    enc = load_condition_encoder(ARGS.enc, device)
    model = load_model(ARGS.weights, device, model_type="auto")

    points = []      # [(target, 生成电荷均值)] → global 拟合
    per_domain = {}
    for i in sel:
        did = str(d["domain_ids"][i])
        pdb_path = Path(ARGS.dompdb) / f"{did}.pdb"
        try:
            protein_dict, *_ = parse_PDB(str(pdb_path), device="cpu", parse_all_atoms=False)
        except Exception as e:
            print(f"  !! {did} parse 失败跳过: {e}", flush=True)
            continue
        L = protein_dict["X"].shape[0]
        native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
        native_q = round(float(net_charge(native, 7.4)))
        protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
        fd = featurize(protein_dict, cutoff_for_score=8.0,
                       model_type="protein_mpnn", use_atom_context=False, number_of_ligand_atoms=0)
        fd["batch_size"] = 1
        fd["temperature"] = ARGS.temperature
        fd["bias"] = torch.zeros(1, L, 21)

        arms = {}
        for off in OFFSETS:
            tgt = native_q + off
            charges = []
            for k in range(ARGS.n_per):
                torch.manual_seed(ARGS.seed * 1000 + i + off + k)
                fd["randn"] = torch.randn(1, L)
                cond_vec = make_condition_vector(7.4, net_charge=tgt)
                out = conditioned_sample(model, enc, fd, cond_vec, device)
                seq = seq_to_string(out["S"][0].cpu().numpy())
                charges.append(float(net_charge(seq, 7.4)))
            mean_q = float(np.mean(charges))
            points.append((tgt, mean_q))
            arms[off] = {"target": tgt, "mean_charge": round(mean_q, 2)}
        per_domain[did] = {"L": L, "native_q": native_q, "arms": arms}
        print(f"{did} (L={L}, native {native_q:+d}): " +
              " ".join(f"{off:+d}→{arms[off]['mean_charge']:+.1f}" for off in OFFSETS), flush=True)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    slope, inter = linfit(xs, ys)
    out = {"global": {"slope": round(slope, 4), "intercept": round(inter, 4),
                      "n_point": len(points), "n_domains": len(per_domain)},
           "per_protein": per_domain}
    with open(ARGS.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n已写 {ARGS.out}: global slope={slope:.3f} intercept={inter:.3f} n_point={len(points)}（{len(per_domain)} 域）", flush=True)


if __name__ == "__main__":
    main()

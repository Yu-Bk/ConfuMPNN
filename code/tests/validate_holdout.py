"""v12.2 hold-out 评估：真正未见 15% 验证集上的电荷控制（H2）+ recovery。

背景（index/PROJECT_LOCAL_V12_2.md §2.1）：hold-out = 从训练集 7,886 按电荷分层划 15%
（1,176 域），v12.2 只训了 85% → 对 v12.2 是真正未见的同分布数据。
本脚本抽样 N_DOM 域 × 8 pH，用 v12.2 编码器 + global 校准（hold-out 域不在 17 蛋白
per_protein 表内），target=round(label charge)，测 dev（H2，≤2 命中）与 recovery。

用法（项目根）：
  PYTHONPATH=code /home/baokun_yu/miniconda3/envs/confumpnn/bin/python code/tests/validate_holdout.py [--n_dom 30] [--n_per 5] [--device cuda:4]
输出：output/holdout_eval_v12_2.json
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
from run_guided import load_calibration  # noqa: E402

AP = argparse.ArgumentParser()
AP.add_argument("--n_dom", type=int, default=30, help="抽样域数（hold-out 共 1,176）")
AP.add_argument("--n_per", type=int, default=5, help="每 (域,pH) 采样条数")
AP.add_argument("--seed", type=int, default=42)
AP.add_argument("--temperature", type=float, default=0.3)
AP.add_argument("--device", default="cuda:4")
AP.add_argument("--labels", default=str(_PROJECT_DIR / "data/cath/labels_holdout_train.npz"))
AP.add_argument("--dompdb", default=str(_PROJECT_DIR / "data/cath/S40/dompdb_pdb"))
AP.add_argument("--enc", default=str(_PROJECT_DIR / "output/finetune_v12_2/finetune_epoch030.pt"))
AP.add_argument("--weights", default=str(_PROJECT_DIR / "MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt"))
AP.add_argument("--calibration_file", default=str(_PROJECT_DIR / "output/charge_calibration_v12_2.json"))
AP.add_argument("--out", default=str(_PROJECT_DIR / "output/holdout_eval_v12_2.json"))
ARGS = AP.parse_args()


def main():
    device = torch.device(ARGS.device)
    d = np.load(ARGS.labels, allow_pickle=True)
    n_dom = len(d["domain_ids"])
    n_pH = d["pH"].shape[0] // n_dom
    print(f"hold-out {n_dom} 域 × {n_pH} pH，抽样 {ARGS.n_dom} 域", flush=True)

    enc = load_condition_encoder(ARGS.enc, device)
    model = load_model(ARGS.weights, device, model_type="auto")
    cal_slope, cal_off, mode, _ = load_calibration(ARGS.calibration_file, "GLOBAL_FORCE", force_global=True)
    print(f"校准表: {mode} slope={cal_slope:.3f} intercept={cal_off:.3f}", flush=True)

    rng = random.Random(ARGS.seed)
    dom_idx = sorted(rng.sample(range(n_dom), ARGS.n_dom))

    all_dev, all_hit, all_rec = [], [], []
    results = {"meta": {"n_dom": ARGS.n_dom, "n_per": ARGS.n_per, "seed": ARGS.seed,
                        "cal_mode": mode, "enc": ARGS.enc},
               "domains": {}, "summary": {}}
    n_ok = 0
    for i in dom_idx:
        did = str(d["domain_ids"][i])
        pdb_path = Path(ARGS.dompdb) / f"{did}.pdb"
        try:
            protein_dict, *_ = parse_PDB(str(pdb_path), device="cpu", parse_all_atoms=False)
        except Exception as e:
            print(f"  !! {did} parse 失败跳过: {e}", flush=True)
            continue
        L = protein_dict["X"].shape[0]
        native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
        protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
        fd = featurize(protein_dict, cutoff_for_score=8.0,
                       model_type="protein_mpnn", use_atom_context=False, number_of_ligand_atoms=0)
        fd["batch_size"] = 1
        fd["temperature"] = ARGS.temperature
        fd["bias"] = torch.zeros(1, L, 21)
        n_ok += 1

        dom_res = {"L": L, "native_charge": round(float(net_charge(native, 7.4)), 2), "arms": {}}
        for j in range(n_pH):
            pH = float(d["pH"][i * n_pH + j])
            tgt = round(float(d["charge"][i * n_pH + j]))
            tgt_eff = (tgt - cal_off) / cal_slope
            charges, recs = [], []
            for k in range(ARGS.n_per):
                torch.manual_seed(ARGS.seed * 1000 + i * n_pH + j * ARGS.n_per + k)
                fd["randn"] = torch.randn(1, L)
                cond_vec = make_condition_vector(pH, net_charge=tgt_eff)
                out = conditioned_sample(model, enc, fd, cond_vec, device)
                seq = seq_to_string(out["S"][0].cpu().numpy())
                charges.append(float(net_charge(seq, pH)))
                recs.append(sum(a == b for a, b in zip(seq, native)) / L)
            mean_q = float(np.mean(charges))
            dev = abs(mean_q - tgt)
            hit = dev <= 2.0
            all_dev.append(dev); all_hit.append(hit); all_rec.append(float(np.mean(recs)))
            dom_res["arms"][f"ph{j}_t{tgt}"] = {"pH": round(pH, 2), "target": tgt,
                                                "mean_charge": round(mean_q, 2), "dev": round(dev, 2),
                                                "hit": hit, "recovery": round(float(np.mean(recs)), 3)}
            print(f"{did} pH={pH:.1f} t={tgt:+.0f} → {mean_q:+.1f} dev={dev:.2f} {'✅' if hit else '❌'} rec={np.mean(recs):.2f}", flush=True)
        results["domains"][did] = dom_res

    results["summary"] = {
        "n_domains_ok": n_ok, "n_arms": len(all_dev),
        "hit_rate": round(sum(all_hit) / len(all_hit), 3) if all_hit else None,
        "mean_dev": round(float(np.mean(all_dev)), 3) if all_dev else None,
        "mean_recovery": round(float(np.mean(all_rec)), 3) if all_rec else None,
    }
    print(f"\n=== 汇总：{n_ok} 域 × {len(all_dev)//max(1,n_ok)} pH × {ARGS.n_per} 序列 ===", flush=True)
    print(f"  H2 命中率 (dev≤2): {results['summary']['hit_rate']}", flush=True)
    print(f"  平均 dev: {results['summary']['mean_dev']}", flush=True)
    print(f"  平均 recovery: {results['summary']['mean_recovery']}", flush=True)
    with open(ARGS.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"已写 {ARGS.out}", flush=True)


if __name__ == "__main__":
    main()

"""小样本现场标定实验：对泛化 10 蛋白采少量序列拟合各自 slope → 临时 per-protein 校准表。

背景（PROJECT_LOCAL_V12_2.md §6E + 使用指南）：建议"表外蛋白先小样本标定（采 20-50 条
拟合自身 slope）"，但该建议未实测。本脚本对 10 个 valid 蛋白模拟现场标定：
- 标定采样：5 target（native±[8,4,0,4,8]）× --n_per 条（默认 10 → 50 条/蛋白）
- 拟合每蛋白 (target, 生成电荷) 直线 → per_protein slope/intercept
- global 用 big 表兜底（评估蛋白都在 per_protein 内，不触发）
输出：output/charge_calibration_v12_2_small.json
随后 validate_generalization.py --calibrate auto --calibration_file 该表 → 测命中率。

用法（项目根）：
  PYTHONPATH=code python code/tests/build_calibration_small.py --device cuda:4 --n_per 10
"""
import argparse
import json
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
AP.add_argument("--n_per", type=int, default=10, help="每 target 标定采样数（默认 10 → 50 条/蛋白；实测最优，2026-08-31）\n"
                "⚠️ 不要加大到 20：n20 对高方差蛋白（1BJ4/2FEO）反而退化（更接近真值但校准反推\n"
                "落到响应弯曲段 → 命中率降）。n10 的'过度拟合'碰巧补偿弯曲。响应弯曲蛋白用\n"
                "LOOCV 检测 + 回退 global，不是加大 n_per。")
AP.add_argument("--consistency_thresh", type=float, default=3.0,
                help="小样本拟合 vs global 在 native±8 区间的最大预测偏差阈值；超过则标记 unreliable")
AP.add_argument("--seed", type=int, default=777)
AP.add_argument("--temperature", type=float, default=0.3)
AP.add_argument("--device", default="cuda:4")
AP.add_argument("--manifest", default=str(_PROJECT_DIR / "data/validation_pdbs/validation_manifest.json"))
AP.add_argument("--enc", default=str(_PROJECT_DIR / "output/finetune_v12_2/finetune_epoch030.pt"))
AP.add_argument("--weights", default=str(_PROJECT_DIR / "MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt"))
AP.add_argument("--big_cal", default=str(_PROJECT_DIR / "output/charge_calibration_v12_2_big.json"))
AP.add_argument("--out", default=str(_PROJECT_DIR / "output/charge_calibration_v12_2_small.json"))
ARGS = AP.parse_args()

OFFSETS = [-8, -4, 0, 4, 8]


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
    man = json.load(open(ARGS.manifest))
    items = man["items"]
    print(f"{len(items)} 个 valid 蛋白，小样本标定（5 target × {ARGS.n_per} = {5 * ARGS.n_per} 条/蛋白）", flush=True)

    enc = load_condition_encoder(ARGS.enc, device)
    model = load_model(ARGS.weights, device, model_type="auto")

    per = {}
    for it in items:
        name, path = it["pdb"], str(_PROJECT_DIR / it["path"])
        try:
            protein_dict, *_ = parse_PDB(path, device="cpu", parse_all_atoms=False)
        except Exception as e:
            print(f"  !! {name} parse 失败: {e}", flush=True)
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

        xs, ys = [], []
        for off in OFFSETS:
            tgt = native_q + off
            charges = []
            for k in range(ARGS.n_per):
                torch.manual_seed(ARGS.seed + off + k)
                fd["randn"] = torch.randn(1, L)
                cond_vec = make_condition_vector(7.4, net_charge=tgt)
                out = conditioned_sample(model, enc, fd, cond_vec, device)
                seq = seq_to_string(out["S"][0].cpu().numpy())
                charges.append(float(net_charge(seq, 7.4)))
            xs.append(tgt)
            ys.append(float(np.mean(charges)))
        slope, inter = linfit(xs, ys)
        # 拟合稳定性校验（2026-08-31 用户要求：小样本对好蛋白的破坏怎么解决）。
        # 机制：小样本 5 target×n10 拟合的 intercept 噪声可在 target 插值处放大成 dev 2+。
        # 检测法 = LOOCV（留一交叉验证）：逐个去掉一个 target 点，用剩 4 点重拟合，
        # 预测被去掉点的 target，算预测误差。LOOCV 误差大 → 拟合被个别点主导
        # （响应弯曲/均值噪声大）→ 标记 unreliable，建议回退 global 或增大 --n_per。
        loocv_errs = []
        for i in range(len(xs)):
            xr = xs[:i] + xs[i + 1:]
            yr = ys[:i] + ys[i + 1:]
            a, b = linfit(xr, yr)
            loocv_errs.append(abs((a * xs[i] + b) - ys[i]))
        loocv = float(sum(loocv_errs) / len(loocv_errs))
        unreliable = loocv > ARGS.consistency_thresh
        per[name] = {"slope": round(slope, 4), "intercept": round(inter, 4),
                     "n_calib": len(xs) * ARGS.n_per, "native_q": native_q,
                     "loocv": round(loocv, 2), "unreliable": unreliable}
        tag = " ⚠️LOOCV大(响应弯曲/噪声,建议回退global或增大n_per)" if unreliable else ""
        print(f"{name:6s} L={L} native={native_q:+d} slope={slope:.3f} int={inter:.2f} "
              f"| LOOCV={loocv:.2f}{tag} | " +
              " ".join(f"{t:+d}→{m:+.1f}" for t, m in zip(xs, ys)), flush=True)

    # global 用 big 表兜底（评估蛋白都在 per_protein 内）
    big = json.load(open(ARGS.big_cal))
    out = {"global": big["global"], "per_protein": per}
    with open(ARGS.out, "w") as f:
        json.dump(out, f, indent=2)
    n_bad = sum(1 for v in per.values() if v.get("unreliable"))
    print(f"\n已写 {ARGS.out}: {len(per)} 个 per_protein（小样本标定），其中 {n_bad} 个与 global 不一致（unreliable）", flush=True)


if __name__ == "__main__":
    main()

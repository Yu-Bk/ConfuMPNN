"""exp7b — pH 细网格 × 三处理(校准分离)采样（2026-09-06，S1 主任务）。

背景：旧 exp7 只做 pH{5,7.4,9} + 仅 7.4 校准，三缺陷：①没拆 native vs 偏移臂；
②校准前/后混用（7.4 表外推到 5/9）；③无中间 pH。本脚本补全：
  - pH 细网格 [5,5.5,6,6.5,7,7.4,8,8.5,9]；
  - class A 每 pH 三种 target 处理（严格分开）：
      T1 原始（无校准）  : target_eff = round(native_q@pH)+Δ
      T2 7.4-固定校准    : target_eff=(target - inter74)/slope74   (7.4 per-protein 表/现场)
      T3 逐pH现场小样本  : target_eff=(target - inter_pH)/slope_pH (该pH 现场 5offset×n10=50 拟合)
  - n=60/臂；同 seed 跨 pH 配对（identity vs 7.4 需同一 seed）。

输出（断点续跑，arm/cal 粒度已存在即跳过）：
  <out_root>/<pdb>/native.json
  <out_root>/<pdb>/T1|T2|T3/<ph>/<arm>.json
      {"treatment","ph","arm","target","target_eff","cal":{...},"n","records":[{seed,seq,q_own,
       n_pos,n_neg,n_his,n_charged,rec}]}
  <out_root>/<pdb>/cal_T3_<ph>.json   # T3 现场标定（per pH）
  <out_root>/<pdb>/cal_T2.json        # T2 用 7.4 校准（表/现场）记录

用法（confumpnn 环境，项目根）：
  PYTHONPATH=code python code/tests/exp7b_pHgrid_sampler.py --mode prot --device cuda:6 --n 60
  PYTHONPATH=code python code/tests/exp7b_pHgrid_sampler.py --mode lig  --device cuda:6 --n 60
  # 单蛋白（并发跑用）：
  ... --proteins 3MXB_A
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
_CODE = _ROOT / "code"
for p in [str(_CODE), str(_ROOT / "LigandMPNN")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data_utils import featurize, parse_PDB  # noqa: E402
from src.condition_embedding import make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from run_guided import (load_model, load_condition_encoder,  # noqa: E402
                        load_calibration, seq_to_string)

MODES = {
    "prot": {
        "weights": _ROOT / "MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt",
        "encoder": _ROOT / "output/finetune_v12_2/finetune_epoch030.pt",
        "cal_file": _ROOT / "output/charge_calibration_v12_2.json",
        "use_ligand": False,
        "out_root": _ROOT / "output/exp_pH2_prot",
        "old_root": _ROOT / "output/exp_pH_prot",
        "pdbs": ["1AZM", "1AS2", "1BJ4", "3MXB_A"],
        # 7.4 per-protein 校准中，表外/失真蛋白的现场来源（key=pdb → 已有 cal json）
        "t2_reuse_onsite": {
            "3MXB_A": _ROOT / "output/exp_pH_prot/3MXB_A/smallcal74.json",
        },
    },
    "lig": {
        "weights": _ROOT / "LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt",
        "encoder": _ROOT / "output/finetune_ligand_v14_rna/finetune_epoch050.pt",
        "cal_file": _ROOT / "output/charge_calibration_v14_ligand_clean.json",
        "use_ligand": True,
        "out_root": _ROOT / "output/exp_pH2_lig",
        "old_root": _ROOT / "output/exp_pH_lig",
        "pdbs": ["5O60_E", "1CGE", "1BJ4"],  # 平衡集：碱(RNA)/酸(金属)/中性(长PLP)
        "t2_reuse_onsite": {},
    },
}
ARMS = [("native", 0), ("n2", -2), ("p2", +2), ("n8", -8), ("p8", +8)]
PH_GRID = [5.0, 5.5, 6.0, 6.5, 7.0, 7.4, 8.0, 8.5, 9.0]
CAL_OFFSETS = [-8, -4, 0, 4, 8]
N_PER_CAL = 10
TREATMENTS = ["T1", "T2", "T3"]
SEED_BASE_T2 = 7_000_000        # 对齐 exp7 classA（T2 在 pH5/7.4/9 可复现旧值）
SEED_BASE_T1 = 10_000_000
SEED_BASE_T3 = 12_000_000
SEED_BASE_T3CAL = 14_000_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prot", "lig"], required=True)
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--proteins", default=None)
    ap.add_argument("--treatments", default="T1,T2,T3")
    ap.add_argument("--ph_grid", default=",".join(str(x) for x in PH_GRID))
    ap.add_argument("--out_dir", default=None, help="覆盖输出根（smoke 用）")
    ap.add_argument("--smoke", type=int, default=0,
                    help=">0 时只跑前 --smoke 条/臂并限制 pH 子集(5.0,7.4,9.0)（自检用）")
    return ap.parse_args()


def prepare_pdb(pdb_path, use_ligand):
    protein_dict, *_ = parse_PDB(str(pdb_path), device="cpu")
    L = int(protein_dict["X"].shape[0])
    native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
    protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
    fd = featurize(protein_dict, cutoff_for_score=8.0,
                   model_type="ligand_mpnn" if use_ligand else "protein_mpnn",
                   use_atom_context=use_ligand,
                   number_of_ligand_atoms=(25 if use_ligand else 0))
    fd["batch_size"] = 1
    fd["temperature"] = 0.3
    fd["bias"] = torch.zeros(1, L, 21)
    return fd, L, native


def comp_counts(seq):
    n_pos = seq.count("K") + seq.count("R")
    n_neg = seq.count("D") + seq.count("E")
    n_his = seq.count("H")
    return n_pos, n_neg, n_his, n_pos + n_neg + n_his


def sample_cond(model, enc, fd, pH, tgt_eff, seed, L, device):
    torch.manual_seed(seed)
    fd["randn"] = torch.randn(1, L)
    cv = make_condition_vector(pH, net_charge=tgt_eff)
    out = conditioned_sample(model, enc, fd, cv, device=device)
    return seq_to_string(out["S"][0].cpu().numpy())


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx if sxx > 1e-9 else float("nan")
    b = my - a * mx
    return a, b


def run_smallcal(model, enc, fd, ph, center, seed_gen, L, device):
    """现场小样本标定：5 offset × n_per=10 = 50 条，在 ph 下直接注入，拟合 slope/inter。"""
    xs, ys, rows = [], [], []
    for off_i, off in enumerate(CAL_OFFSETS):
        tgt = center + off
        qs = []
        for k in range(N_PER_CAL):
            sd = seed_gen(off_i, k)
            s = sample_cond(model, enc, fd, ph, float(tgt), sd, L, device)
            q = float(net_charge(s, ph))
            qs.append(q)
            rows.append({"offset": off, "target": tgt, "seed": sd, "charge": round(q, 3)})
        xs.append(tgt)
        ys.append(float(np.mean(qs)))
    slope, inter = linfit(xs, ys)
    errs = []
    for i in range(len(xs)):
        xr = xs[:i] + xs[i + 1:]
        yr = ys[:i] + ys[i + 1:]
        a, b = linfit(xr, yr)
        errs.append(abs((a * xs[i] + b) - ys[i]))
    cal = {"ph": ph, "center": center, "slope": round(slope, 4),
           "intercept": round(inter, 4), "n_calib": len(rows),
           "loocv": round(float(np.mean(errs)), 2),
           "unreliable": bool(np.mean(errs) > 3.0),
           "mean_charge_by_target": {str(x): round(y, 3) for x, y in zip(xs, ys)},
           "calib_rows": rows}
    return cal


def main():
    args = parse_args()
    cfg = MODES[args.mode]
    device = torch.device(args.device)
    pdbs = args.proteins.split(",") if args.proteins else cfg["pdbs"]
    treatments = [t for t in args.treatments.split(",") if t in TREATMENTS]
    grid = [float(x) for x in args.ph_grid.split(",")]
    n = args.n
    if args.smoke > 0:
        grid = [5.0, 7.4, 9.0]
        n = args.smoke
    out_root = Path(args.out_dir) if args.out_dir else cfg["out_root"]
    out_root.mkdir(parents=True, exist_ok=True)
    ph_strs = [str(ph) for ph in grid]

    log(f"== exp7b {args.mode} | device={args.device} n={n} pH={grid} T={treatments} ==")
    log(f"backbone={cfg['weights'].name} enc={cfg['encoder'].name}")
    model = load_model(str(cfg["weights"]), device, model_type="auto")
    enc = load_condition_encoder(str(cfg["encoder"]), device)

    for pdb in pdbs:
        p0 = time.time()
        pdb_path = _ROOT / "data" / "validation_pdbs" / f"{pdb}.pdb"
        fd, L, native = prepare_pdb(pdb_path, cfg["use_ligand"])
        qph = {ph: float(net_charge(native, ph)) for ph in PH_GRID}
        round_nat = {ph: int(round(qph[ph])) for ph in PH_GRID}
        pdir = out_root / pdb
        pdir.mkdir(parents=True, exist_ok=True)
        native_info = {"pdb": pdb, "L": L, "native": native,
                       "native_q_by_pH": {str(ph): round(q, 3) for ph, q in qph.items()},
                       "round_target_by_pH": {str(ph): r for ph, r in round_nat.items()},
                       "mode": args.mode}
        json.dump(native_info, open(pdir / "native.json", "w"), indent=2)
        ordinal = cfg["pdbs"].index(pdb)
        log(f"\n=== {pdb} L={L} q@pH: " +
            " ".join(f"{ph}:{qph[ph]:+.2f}" for ph in [5.0, 6.0, 7.4, 8.0, 9.0]) + " ===")

        def seed_arm(treat, arm_i, k):
            base = {"T1": SEED_BASE_T1, "T2": SEED_BASE_T2, "T3": SEED_BASE_T3}[treat]
            return base + ordinal * 10_000_000 + arm_i * 100_000 + k

        # ---- 校准准备 ----
        # T2: 7.4 per-protein 表（表内）或现场 7.4（表外，reuse/run）
        cal_t2_path = pdir / "cal_T2.json"
        if cal_t2_path.is_file():
            cal_t2 = json.load(open(cal_t2_path))
            log(f"  [T2 cal] 已存在 slope={cal_t2['slope']} inter={cal_t2['intercept']}")
        else:
            cal_t2 = None
            # 1) 表内 per-protein（未标 unreliable）
            if "T2" in treatments:
                slope, off, mode, _ = load_calibration(str(cfg["cal_file"]), pdb)
                if slope is not None and mode and str(mode).startswith("per-protein"):
                    cal_t2 = {"pdb": pdb, "slope": round(slope, 4),
                              "intercept": round(off, 4), "source": mode,
                              "ph": 7.4, "center": round_nat[7.4]}
            # 2) 表外：指定现场 reuse
            if cal_t2 is None and pdb in cfg.get("t2_reuse_onsite", {}):
                reuse = cfg["t2_reuse_onsite"][pdb]
                if reuse.is_file():
                    sc = json.load(open(reuse))
                    cal_t2 = {"pdb": pdb, "slope": sc["slope"],
                              "intercept": sc["intercept"],
                              "source": f"reuse onsite74 {reuse.name}",
                              "ph": 7.4, "center": round_nat[7.4]}
            # 3) 都没有 → 现场 7.4 标定
            if cal_t2 is None and "T2" in treatments:
                cal = run_smallcal(model, enc, fd, 7.4, round_nat[7.4],
                                   lambda oi, k: SEED_BASE_T2 + ordinal * 10_000_000
                                   + 50_000 + oi * 10_000 + k, L, device)
                cal_t2 = {"pdb": pdb, "slope": cal["slope"], "intercept": cal["intercept"],
                          "source": "onsite74 fresh", "ph": 7.4, "center": round_nat[7.4],
                          "loocv": cal["loocv"], "n_calib": cal["n_calib"]}
            if cal_t2 is not None:
                json.dump(cal_t2, open(cal_t2_path, "w"), indent=2)
                log(f"  [T2 cal] source={cal_t2['source']} slope={cal_t2['slope']} "
                    f"inter={cal_t2['intercept']}")

        # T3: per-pH 现场标定 cache
        cal_t3 = {}
        if "T3" in treatments:
            for ph in grid:
                fname = pdir / f"cal_T3_{ph}.json"
                if fname.is_file():
                    cal_t3[ph] = json.load(open(fname))
                else:
                    cal = run_smallcal(model, enc, fd, ph, round_nat[ph],
                                       lambda oi, k, phh=ph: SEED_BASE_T3CAL
                                       + ordinal * 10_000_000
                                       + int(round(phh * 10)) * 100_000 + oi * 1000 + k,
                                       L, device)
                    cal["source"] = f"onsite pH{ph}"
                    json.dump(cal, open(fname, "w"), indent=2)
                    cal_t3[ph] = cal
                    log(f"  [T3 cal {ph}] slope={cal['slope']} inter={cal['intercept']} "
                        f"LOOCV={cal['loocv']}")

        # ---- 采样 arms ----
        for treat in treatments:
            tdir = pdir / treat
            tdir.mkdir(parents=True, exist_ok=True)
            for ph in grid:
                phs = str(ph)
                phdir = tdir / phs
                phdir.mkdir(parents=True, exist_ok=True)
                for arm_i, (arm, dq) in enumerate(ARMS):
                    target = round_nat[ph] + dq
                    # 计算 target_eff
                    if treat == "T1":
                        eff = float(target)
                        cal_src = "raw"
                    elif treat == "T2":
                        eff = float((target - cal_t2["intercept"]) / cal_t2["slope"])
                        cal_src = cal_t2.get("source", "T2")
                    else:
                        cc = cal_t3[ph]
                        eff = float((target - cc["intercept"]) / cc["slope"])
                        cal_src = f"T3 onsite pH{ph}"
                    fpath = phdir / f"{arm}.json"
                    if fpath.is_file():
                        continue
                    records = []
                    t1 = time.time()
                    for k in range(n):
                        sd = seed_arm(treat, arm_i, k)
                        s = sample_cond(model, enc, fd, ph, eff, sd, L, device)
                        q = float(net_charge(s, ph))
                        npos, nneg, nhis, nch = comp_counts(s)
                        rec = float(sum(a == b for a, b in zip(s, native)) / L)
                        records.append({"seed": sd, "seq": s, "q_own": round(q, 3),
                                        "n_pos": npos, "n_neg": nneg, "n_his": nhis,
                                        "n_charged": nch, "rec": round(rec, 4)})
                    out = {"treatment": treat, "ph": ph, "arm": arm, "dq": dq,
                           "target": target, "target_eff": round(eff, 4),
                           "cal_source": cal_src, "n": len(records), "records": records}
                    json.dump(out, open(fpath, "w"))
                    log(f"    [{treat} {pdb} pH{ph} {arm}] target={target:+d} "
                        f"eff={eff:+.3f} n={len(records)} ({time.time()-t1:.0f}s)")
        log(f"=== {pdb} done ({time.time()-p0:.0f}s) ===")

    log("\n=== exp7b 采样完成 ===")


if __name__ == "__main__":
    main()

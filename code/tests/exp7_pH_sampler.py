"""exp7 — 跨 pH(5/7.4/9) pH 敏感性采样（2026-09-06）。GPU=cuda:6。

设计（见 compare/plan_exp7_pH_response.md）：
  class A（换 pH 看臂）: 对每蛋白在 pH∈{5,7.4,9} 下各做 5 臂 native/n2/p2/n8/p8 采样，
      target = round(native_q@pH)+Δ（pH 锚定）；注入按 **7.4 校准表**（表内 per-protein /
      表外小样本 onsite 标定）。同 seed 跨 pH → 配对可比。量 dev/H2/per-seq 命中。
  class B（固定 target 换 pH）: 固定 target（T0=round(native_q@7.4) 与 T1=0），在 pH 5/7.4/9
      采样，**不校准**直接注入；看实测净电荷(net_charge@本pH) 与序列是否随 pH 变（identity）。

蛋白模式：MoMPNN + v12.2 编码器；配体模式：LigandMPNN + v14 编码器（atom_context=25）。

校准来源（class A）：
  蛋白：1AZM/1AS2 用 output/charge_calibration_v12_2.json per-protein；
        1BJ4 复用 output/exp_control_prot/1BJ4/route_Bsmall/1BJ4_smallcal.json（现场 7.4 标定）；
        3MXB_A（表外）现场小样本标定（5 offset×10=50 条，@7.4）。
  配体：5O60_E/1CGE/2FEO 用 output/charge_calibration_v14_ligand_clean.json per-protein。

输出（断点续跑：已完成 (class,pdb) 跳过）：
  output/exp_pH_prot/{pdb}/native.json
  output/exp_pH_prot/{pdb}/classA.json   # {"pH":{"arm":{"target":,"target_eff":,"n":,"records":[...]}}}
  output/exp_pH_prot/{pdb}/classB.json   # {"target":int,"pH":{"n":,"records":[...]}}
  （配体模式同理 output/exp_pH_lig/）
records: {seed, seq, q_own(净电荷@本pH), n_pos(K+R), n_neg(D+E), n_his(H), n_charged,
          rec(相对 native 一致率)}

用法（项目根 confumpnn 环境）：
  PYTHONPATH=code python code/tests/exp7_pH_sampler.py --mode prot --device cuda:6 --n 60
  PYTHONPATH=code python code/tests/exp7_pH_sampler.py --mode lig  --device cuda:6 --n 60
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

# ---------- 模式设定 ----------
MODES = {
    "prot": {
        "weights": _ROOT / "MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt",
        "encoder": _ROOT / "output/finetune_v12_2/finetune_epoch030.pt",
        "cal_file": _ROOT / "output/charge_calibration_v12_2.json",
        "use_ligand": False,
        "out_root": _ROOT / "output/exp_pH_prot",
        "pdbs": ["1AZM", "1AS2", "1BJ4", "3MXB_A"],
        # per-pdb cal override（可空）："slope": , "intercept":, "source":, "reuse_smallcal": path
        "override_cal": {
            "1BJ4": {"reuse": _ROOT / "output/exp_control_prot/1BJ4/route_Bsmall/1BJ4_smallcal.json",
                     "source": "exp1 route_Bsmall onsite cal @7.4"},
        },
    },
    "lig": {
        "weights": _ROOT / "LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt",
        "encoder": _ROOT / "output/finetune_ligand_v14_rna/finetune_epoch050.pt",
        "cal_file": _ROOT / "output/charge_calibration_v14_ligand_clean.json",
        "use_ligand": True,
        "out_root": _ROOT / "output/exp_pH_lig",
        "pdbs": ["5O60_E", "1CGE", "2FEO"],
        "override_cal": {},
    },
}
ARMS = [("native", 0), ("n2", -2), ("p2", +2), ("n8", -8), ("p8", +8)]
ARMS_DQ = dict(ARMS)
PHIS = [5.0, 7.4, 9.0]
CAL_OFFSETS = [-8, -4, 0, 4, 8]
N_PER_CAL = 10
SEED_BASE_A = 7000000          # class A seed base
SEED_BASE_B = 8000000          # class B seed base


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prot", "lig"], required=True)
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--proteins", default=None)
    ap.add_argument("--classes", default="A,B")
    ap.add_argument("--out_dir", default=None, help="覆盖输出根（smoke 用）")
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
    qph = {ph: float(net_charge(native, ph)) for ph in PHIS}
    return fd, L, native, qph


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


def onsite_smallcal(model, enc, fd, pdb, round74, L, device, cache_path):
    """现场小样本标定 @7.4：5 offset × n_per=10（复制 exp1 Bsmall 协议）。"""
    if cache_path.is_file():
        sc = json.load(open(cache_path))
        log(f"  [cal] 复用小样本标定 {cache_path.name} slope={sc['slope']:.3f} "
            f"inter={sc['intercept']:.2f}")
        return sc["slope"], sc["intercept"], "per-protein(small-sample@7.4)"
    xs, ys = [], []
    for off in CAL_OFFSETS:
        tgt = round74 + off
        qs = []
        for k in range(N_PER_CAL):
            sd = SEED_BASE_A + 50000 + off * 100 + k
            s = sample_cond(model, enc, fd, 7.4, float(tgt), sd, L, device)
            qs.append(float(net_charge(s, 7.4)))
        xs.append(tgt)
        ys.append(float(np.mean(qs)))
    slope, inter = linfit(xs, ys)
    errs = []
    for i in range(len(xs)):
        xr = xs[:i] + xs[i + 1:]
        yr = ys[:i] + ys[i + 1:]
        a, b = linfit(xr, yr)
        errs.append(abs((a * xs[i] + b) - ys[i]))
    sc = {"slope": round(slope, 4), "intercept": round(inter, 4),
          "n_calib": len(xs) * N_PER_CAL, "loocv": round(float(np.mean(errs)), 2),
          "unreliable": bool(np.mean(errs) > 3.0),
          "native_q74": round74}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(sc, open(cache_path, "w"), indent=2)
    log(f"  [cal] 现场小样本标定 {cache_path.name} slope={slope:.3f} inter={inter:.2f} "
        f"LOOCV={sc['loocv']:.2f}")
    return slope, inter, "per-protein(small-sample@7.4)"


def get_cal_slope(model, enc, fd, pdb, cal_file, round74, L, device, out_root, ovr):
    """返回 (slope, intercept, source)。ovr 为 override 映射（mode 级）。"""
    if pdb in ovr:
        o = ovr[pdb]
        if "reuse" in o and o["reuse"].is_file():
            sc = json.load(open(o["reuse"]))
            return sc["slope"], sc["intercept"], o["source"]
        if "slope" in o:
            return o["slope"], o["intercept"], o["source"]
    slope, off, mode, _ = load_calibration(str(cal_file), pdb)
    if slope is not None and mode and str(mode).startswith("per-protein"):
        return slope, off, f"per-protein({pdb})@table"
    # 表内无可用的 per-protein（含 unreliable 回退 global）或表外 → 现场小样本标定 @7.4
    cache = out_root / pdb / "smallcal74.json"
    return onsite_smallcal(model, enc, fd, pdb, round74, L, device, cache)


def main():
    args = parse_args()
    cfg = MODES[args.mode]
    device = torch.device(args.device)
    pdbs = args.proteins.split(",") if args.proteins else cfg["pdbs"]
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    n = args.n
    out_root = Path(args.out_dir) if args.out_dir else cfg["out_root"]
    out_root.mkdir(parents=True, exist_ok=True)

    log(f"== exp7 {args.mode} mode | device={args.device} n={n} pH={PHIS} ==")
    log(f"backbone={cfg['weights'].name} enc={cfg['encoder'].name}")
    model = load_model(str(cfg["weights"]), device, model_type="auto")
    enc = load_condition_encoder(str(cfg["encoder"]), device)

    for p_idx, pdb in enumerate(pdbs):
        t0 = time.time()
        pdb_path = _ROOT / "data" / "validation_pdbs" / f"{pdb}.pdb"
        fd, L, native, qph = prepare_pdb(pdb_path, cfg["use_ligand"])
        round74 = int(round(qph[7.4]))
        pdir = out_root / pdb
        pdir.mkdir(parents=True, exist_ok=True)
        native_info = {"pdb": pdb, "L": L, "native": native,
                       "native_q_by_pH": {str(ph): round(q, 3) for ph, q in qph.items()},
                       "round_target_74": round74}
        json.dump(native_info, open(pdir / "native.json", "w"), indent=2)
        log(f"\n=== {pdb} L={L} q5={qph[5.0]:+.2f} q74={qph[7.4]:+.2f} "
            f"q9={qph[9.0]:+.2f} round74={round74:+d} ===")

        # 校准（class A）
        slope = off = None
        if "A" in classes:
            slope, off, cal_src = get_cal_slope(model, enc, fd, pdb, cfg["cal_file"],
                                                round74, L, device, out_root,
                                                cfg.get("override_cal", {}))
            log(f"  classA cal source: {cal_src} slope={slope:.3f} inter={off:.3f}")

        # ---- class A ----
        if "A" in classes:
            ca_path = pdir / "classA.json"
            if ca_path.is_file():
                log("  classA 已存在，跳过")
            else:
                ca = {"pdb": pdb, "n": n, "cal_source": cal_src,
                      "slope": round(slope, 4), "intercept": round(off, 4),
                      "note": "target=round(native_q@pH)+dq; 注入经 7.4 校准; "
                              "同 seed 跨 pH 配对",
                      "by_pH": {}}
                for ph in PHIS:
                    phs = str(ph)
                    tgt_nat = int(round(qph[ph]))
                    ca["by_pH"][phs] = {}
                    for arm, dq in ARMS:
                        tgt = tgt_nat + dq
                        tgt_eff = float((tgt - off) / slope)
                        records = []
                        for k in range(n):
                            # seed: p_idx*1e7 + arm_index*1e5 + k (pH 无关 → 跨 pH 配对)
                            arm_i = [a for a, _ in ARMS].index(arm)
                            sd = SEED_BASE_A + p_idx * 10_000_000 + arm_i * 100_000 + k
                            s = sample_cond(model, enc, fd, ph, tgt_eff, sd, L, device)
                            q = float(net_charge(s, ph))
                            npos, nneg, nhis, nch = comp_counts(s)
                            rec = float(sum(a == b for a, b in zip(s, native)) / L)
                            records.append({"seed": sd, "seq": s, "q_own": round(q, 3),
                                            "n_pos": npos, "n_neg": nneg, "n_his": nhis,
                                            "n_charged": nch, "rec": round(rec, 4)})
                        ca["by_pH"][phs][arm] = {"target": tgt,
                                                 "target_eff": round(tgt_eff, 4),
                                                 "n": len(records), "records": records}
                        log(f"    [A {pdb} pH{ph} {arm}] tgt={tgt:+d} "
                            f"n={len(records)}")
                json.dump(ca, open(ca_path, "w"), indent=1)
                log(f"  classA → {ca_path} ({time.time()-t0:.0f}s)")

        # ---- class B ----
        if "B" in classes:
            cb_path = pdir / "classB.json"
            if cb_path.is_file():
                log("  classB 已存在，跳过")
            else:
                targets = [("T_native74", round74), ("T_zero", 0)]
                cb = {"pdb": pdb, "n": n,
                      "note": "固定 target 跨 pH（不校准直接注入）; 同 seed 跨 pH 配对; "
                              "q_own=net_charge(seq, 本 pH)",
                      "targets": {}}
                for tidx, (tlab, tgt) in enumerate(targets):
                    cb["targets"][tlab] = {"target": tgt, "pH": {}}
                    for ph in PHIS:
                        phs = str(ph)
                        records = []
                        for k in range(n):
                            sd = SEED_BASE_B + p_idx * 10_000_000 + tidx * 100_000 + k
                            s = sample_cond(model, enc, fd, ph, float(tgt), sd, L, device)
                            q = float(net_charge(s, ph))
                            npos, nneg, nhis, nch = comp_counts(s)
                            rec = float(sum(a == b for a, b in zip(s, native)) / L)
                            records.append({"seed": sd, "seq": s, "q_own": round(q, 3),
                                            "n_pos": npos, "n_neg": nneg, "n_his": nhis,
                                            "n_charged": nch, "rec": round(rec, 4)})
                        cb["targets"][tlab]["pH"][phs] = {"n": len(records),
                                                          "records": records}
                        log(f"    [B {pdb} {tlab}(tgt={tgt:+d}) pH{ph}] n={len(records)}")
                json.dump(cb, open(cb_path, "w"), indent=1)
                log(f"  classB → {cb_path} ({time.time()-t0:.0f}s)")

    log("\n=== exp7 采样完成 ===")


if __name__ == "__main__":
    main()

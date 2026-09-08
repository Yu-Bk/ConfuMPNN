"""exp_control_prot — 蛋白模式对照实验采样驱动（exp1 route A/B + exp2 route C + B 小样本附加组）。

设计（见 compare/plan_exp1_barebackbone_control.md + ablation/plan_exp2_bias_vs_encoder.md）：
  route A（bare） : MoMPNN 裸 backbone 无条件重设计 n=1000/蛋白（无 encoder 无 bias）
  route B（cond） : MoMPNN + v12.2 ConditionEncoder 条件注入，5 臂 native/n2/p2/n8/p8 × n=1000
                    默认校准 = run_guided 同款（auto 读 charge_calibration_v12_2.json，per-protein）
  route Bsmall   : 附加组——现场小样本标定（5 offset × n_per=10=50 条拟合自身 slope/intercept，
                    再以该 per-protein 校准重采样 5 臂）
  route C（bias）: 去掉 encoder，仅 guided_sampler 电荷引导 logit bias（ChargeLookahead strength=0.5）

同 seed 体系：seed(p_idx, arm_idx, k) = SEED_BASE + p_idx*1_000_000 + arm_idx*10_000 + k
  route A 用 arm_idx=native 那组 seed → 与 route B native 臂逐条配对可比。

输出（断点续跑安全；per_seq.jsonl 为追加，fasta/summary 每次由完整 jsonl 重建）：
  output/exp_control_prot/meta.json
  output/exp_control_prot/{pdb}/meta.json
  output/exp_control_prot/{pdb}/route_{A,B,Bsmall,C}/seqs.fa
  output/exp_control_prot/{pdb}/route_{A,B,Bsmall,C}/per_seq.jsonl
  output/exp_control_prot/{pdb}/route_{A,B,Bsmall,C}/summary.json
  output/exp_control_prot/{pdb}/route_Bsmall/{pdb}_smallcal.json

用法（项目根，confumpnn 环境）：
  PYTHONPATH=code python code/tests/exp_prot_control/sample_control.py \
      --mode A --device cuda:2 --n 1000 --proteins 1AZM,1AS2,1BJ4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
_CODE = _PROJECT / "code"
for p in [str(_CODE), str(_PROJECT / "LigandMPNN")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data_utils import featurize, parse_PDB  # noqa: E402
from src.charge_lookahead import make_dynamic_callback  # noqa: E402
from src.condition_embedding import make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from src.guided_sampler import GuidedSampler  # noqa: E402
from run_guided import (load_calibration, load_condition_encoder,  # noqa: E402
                        load_model, seq_to_string)

WEIGHTS = _PROJECT / "MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt"
ENC = _PROJECT / "output/finetune_v12_2/finetune_epoch030.pt"
CAL_TABLE = _PROJECT / "output/charge_calibration_v12_2.json"
pH = 7.4
TEMP = 0.3
BIAS_STRENGTH = 0.5
ARMS = [("native", 0), ("n2", -2), ("p2", +2), ("n8", -8), ("p8", +8)]
ARM_IDX = {name: i for i, (name, _) in enumerate(ARMS)}
SEED_BASE = 424242
CALIB_OFFSETS = [-8, -4, 0, 4, 8]
DEFAULT_OUT = _PROJECT / "output/exp_control_prot"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B", "Bsmall", "C"], required=True)
    ap.add_argument("--proteins", default="1AZM,1AS2,1BJ4")
    ap.add_argument("--n", type=int, default=None,
                    help="每(蛋白×臂)采样数：A/B/Bsmall 默认 1000，C 默认 300")
    ap.add_argument("--arms", default=None)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--seed_base", type=int, default=SEED_BASE)
    ap.add_argument("--n_per_calib", type=int, default=10)
    ap.add_argument("--pdb_map", default=None,
                    help="pdb=path,pdb=path ... 覆盖默认 data/validation_pdbs/<pdb>.pdb 的取数位置"
                         "（如补入蛋白 1LYZ=data/transfer_test/1LYZ.pdb）")
    return ap.parse_args()


def featurize_pdb(path):
    protein_dict, *_ = parse_PDB(str(path), device="cpu")
    L = int(protein_dict["X"].shape[0])
    native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
    protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
    fd = featurize(protein_dict, cutoff_for_score=8.0,
                   model_type="protein_mpnn", use_atom_context=False,
                   number_of_ligand_atoms=0)
    fd["batch_size"] = 1
    fd["temperature"] = TEMP
    fd["bias"] = torch.zeros(1, L, 21)
    return fd, L, native, float(net_charge(native, pH))


def seed_for(p_idx, arm_idx, k, base):
    return base + p_idx * 1_000_000 + arm_idx * 10_000 + k


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx if sxx > 1e-9 else float("nan")
    b = my - a * mx
    return a, b


def sample_cond(model, enc, fd, tgt_eff, seed, L, device):
    torch.manual_seed(seed)
    fd["randn"] = torch.randn(1, L)
    cv = make_condition_vector(pH, net_charge=tgt_eff)
    out = conditioned_sample(model, enc, fd, cv, device=device)
    return seq_to_string(out["S"][0].cpu().numpy())


def sample_bare(model, fd, seed, L, device):
    torch.manual_seed(seed)
    fd["randn"] = torch.randn(1, L)
    cv = make_condition_vector(pH)  # unused when enc=None
    out = conditioned_sample(model, None, fd, cv, device=device)
    return seq_to_string(out["S"][0].cpu().numpy())


def sample_bias(model, sampler, fd, tgt, seed, L, device):
    cb = make_dynamic_callback(pH=pH, target_charge=tgt,
                               structure_filter=None, strength=BIAS_STRENGTH)
    torch.manual_seed(seed)
    fd["randn"] = torch.randn(1, L)
    out = sampler.sample(fd, bias_callback=cb)
    return seq_to_string(out["S"][0].cpu().numpy())


def rec(s, native):
    return float(sum(a == b for a, b in zip(s, native)) / len(native))


def rebuild_fasta_and_summary(jsonl_path, fasta_path, summ_path, qnat, native,
                              route, pdb, n_expected, per_arm=True):
    """从完整 per_seq.jsonl 重建 seqs.fa 与 summary（断点续跑安全）。"""
    rows = [json.loads(line) for line in open(jsonl_path)]
    if per_arm:
        if len(rows) < n_expected:
            return None  # 未完成，不重建
    with open(fasta_path, "w") as f:
        for r in rows:
            if r.get("target") is not None:
                f.write(f">seed_{r['seed']} arm={r['arm']} target={r['target']:+d} "
                        f"q={r['charge']:+.2f}\n{r['seq']}\n")
            else:
                f.write(f">seed_{r['seed']} arm={r['arm']} q={r['charge']:+.2f}\n"
                        f"{r['seq']}\n")
        f.write(f">native charge={qnat:+.2f}\n{native}\n")
    qs = np.array([r["charge"] for r in rows])
    tgt = rows[0]["target"] if per_arm and rows else None
    devs = np.abs(qs - tgt) if tgt is not None else None
    summ = {
        "pdb": pdb, "route": route,
        "n": len(rows),
        "mean_charge": round(float(qs.mean()), 3),
        "std_charge": round(float(qs.std()), 3),
        "recovery_mean": round(float(np.mean([r["recovery"] for r in rows])), 4),
    }
    if per_arm and rows:
        arm = rows[0]["arm"]
        summ["arm"] = arm
        summ["target"] = int(tgt)
        summ["target_eff"] = rows[0].get("target_eff")
        # 均值口径 dev = |mean_q − target|（与既有 H2 验证约定一致）
        summ["dev_of_mean"] = round(float(abs(qs.mean() - tgt)), 3)
        summ["mean_abs_dev"] = round(float(devs.mean()), 3)  # 逐序列平均绝对偏离（分布弥散）
        summ["arm_H2_hit_dev_of_mean_le2"] = bool(abs(qs.mean() - tgt) <= 2.0)
        summ["per_seq_hit_le2"] = round(float((devs <= 2.0).mean()), 4)
    with open(summ_path, "w") as f:
        json.dump(summ, f, indent=2)
    return summ


def main():
    args = parse_args()
    device = torch.device(args.device)
    proteins = args.proteins.split(",")
    pdb_map = {}
    if args.pdb_map:
        for kv in args.pdb_map.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                pdb_map[k.strip()] = v.strip()
    n = args.n if args.n is not None else (1000 if args.mode in ("A", "B", "Bsmall") else 300)
    arm_names = [a for a in (args.arms or "native,n2,p2,n8,p8").split(",")]
    arms = [(a, dict(ARMS)[a]) for a in arm_names if a in dict(ARMS)]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[load] model ... device={device}", flush=True)
    model = load_model(str(WEIGHTS), device, model_type="auto")
    enc = None
    if args.mode in ("B", "Bsmall"):
        print("[load] condition encoder ...", flush=True)
        enc = load_condition_encoder(str(ENC), device)
    sampler = GuidedSampler(model, device=device)

    for p_idx, pdb in enumerate(proteins):
        t0 = time.time()
        pdb_file = (_PROJECT / pdb_map[pdb]) if pdb in pdb_map else (
            _PROJECT / "data/validation_pdbs" / f"{pdb}.pdb")
        fd, L, native, qnat = featurize_pdb(pdb_file)
        if pdb in pdb_map:
            print(f"[pdb_map] {pdb} → {pdb_file}", flush=True)
        t_native = int(round(qnat))
        print(f"\n=== {pdb} L={L} native_q={qnat:+.2f} round={t_native} "
              f"({args.mode}, n={n}) ===", flush=True)

        pdb_dir = out_root / pdb
        pdb_dir.mkdir(parents=True, exist_ok=True)
        meta = {"pdb": pdb, "L": L, "native": native,
                "native_charge": round(qnat, 3), "round_target": t_native,
                "pdb_file": str(pdb_file)}
        cal = None
        if args.mode in ("B", "Bsmall"):
            if args.mode == "B":
                slope, off, mode_lab, _ = load_calibration(str(CAL_TABLE), pdb)
                if slope is None:
                    slope, off, mode_lab, _ = load_calibration(str(CAL_TABLE), pdb, force_global=True)
                cal = (slope, off, mode_lab)
                meta["calibration"] = {"mode": mode_lab, "slope": slope,
                                       "intercept": off, "source": str(CAL_TABLE)}
            else:
                small_json = pdb_dir / "route_Bsmall" / f"{pdb}_smallcal.json"
                if small_json.is_file():
                    sc = json.load(open(small_json))
                    cal = (sc["slope"], sc["intercept"], "per-protein(small-sample)")
                    print(f"  复用小样本标定: slope={sc['slope']:.3f} inter={sc['intercept']:.2f}",
                          flush=True)
                else:
                    cal_dir = pdb_dir / "route_Bsmall"
                    cal_dir.mkdir(parents=True, exist_ok=True)
                    xs, ys, calib_rows = [], [], []
                    for off in CALIB_OFFSETS:
                        tgt = t_native + off
                        qs = []
                        for k in range(args.n_per_calib):
                            sd = seed_for(p_idx, ARM_IDX["native"], 50000 + k + off, args.seed_base)
                            s = sample_cond(model, enc, fd, float(tgt), sd, L, device)
                            q = float(net_charge(s, pH))
                            qs.append(q)
                            calib_rows.append({"offset": off, "target": tgt, "seed": sd,
                                               "charge": round(q, 3)})
                        xs.append(tgt)
                        ys.append(float(np.mean(qs)))
                    slope, inter = linfit(xs, ys)
                    errs = []
                    for i in range(len(xs)):
                        xr = xs[:i] + xs[i + 1:]
                        yr = ys[:i] + ys[i + 1:]
                        a, b = linfit(xr, yr)
                        errs.append(abs((a * xs[i] + b) - ys[i]))
                    loocv = float(np.mean(errs))
                    sc = {"slope": round(slope, 4), "intercept": round(inter, 4),
                          "n_calib": len(xs) * args.n_per_calib, "native_q": t_native,
                          "loocv": round(loocv, 2), "unreliable": bool(loocv > 3.0),
                          "calib_rows": calib_rows,
                          "mean_charge_by_target": {str(t): round(m, 3)
                                                    for t, m in zip(xs, ys)}}
                    with open(small_json, "w") as f:
                        json.dump(sc, f, indent=2)
                    print(f"  小样本标定: slope={slope:.3f} inter={inter:.2f} "
                          f"LOOCV={loocv:.2f} → {small_json.name}", flush=True)
                    cal = (slope, inter, "per-protein(small-sample)")
                meta["calibration"] = {"mode": cal[2], "slope": cal[0],
                                       "intercept": cal[1], "source": "small-sample on-site"}
        with open(pdb_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        route_dir = pdb_dir / f"route_{args.mode}"
        route_dir.mkdir(parents=True, exist_ok=True)

        if args.mode == "A":
            out_path = route_dir / "per_seq.jsonl"
            done = sum(1 for _ in open(out_path)) if out_path.is_file() else 0
            if done < n:
                fw = open(out_path, "a")
                for k in range(done, n):
                    sd = seed_for(p_idx, ARM_IDX["native"], k, args.seed_base)
                    s = sample_bare(model, fd, sd, L, device)
                    q = float(net_charge(s, pH))
                    fw.write(json.dumps({"seed": sd, "seq": s, "charge": round(q, 3),
                                         "arm": "pool", "target": None,
                                         "recovery": round(rec(s, native), 4)}) + "\n")
                    if (k + 1) % 200 == 0:
                        fw.flush()
                        print(f"  A[{k+1}/{n}] ...", flush=True)
                fw.close()
            summ = rebuild_fasta_and_summary(out_path, route_dir / "seqs.fa",
                                             route_dir / "arm_summary.json",
                                             qnat, native, "A", pdb, n, per_arm=False)
            # 兼容：summary.json 顶层
            with open(route_dir / "summary.json", "w") as f:
                json.dump({"pdb": pdb, "mode": "A", "n": n,
                           "native_charge": round(qnat, 3),
                           "pool": summ}, f, indent=2, ensure_ascii=False)
            print(f"  route A done: {summ['n']} seq ({time.time()-t0:.0f}s)", flush=True)
        else:
            arm_summaries = {}
            for a_idx, (arm, dq) in enumerate(arms):
                tgt = t_native + dq
                if args.mode in ("B", "Bsmall"):
                    tgt_eff = float((tgt - cal[1]) / cal[0]) if cal is not None else float(tgt)
                else:
                    tgt_eff = None
                arm_dir = route_dir / f"arm_{arm}"
                arm_dir.mkdir(parents=True, exist_ok=True)
                out_path = arm_dir / "per_seq.jsonl"
                done = sum(1 for _ in open(out_path)) if out_path.is_file() else 0
                if done < n:
                    fw = open(out_path, "a")
                    for k in range(done, n):
                        sd = seed_for(p_idx, a_idx, k, args.seed_base)
                        if args.mode in ("B", "Bsmall"):
                            s = sample_cond(model, enc, fd, tgt_eff, sd, L, device)
                        else:
                            s = sample_bias(model, sampler, fd, tgt, sd, L, device)
                        q = float(net_charge(s, pH))
                        fw.write(json.dumps({"seed": sd, "seq": s, "charge": round(q, 3),
                                             "arm": arm, "target": tgt,
                                             "target_eff": (None if tgt_eff is None
                                                            else round(tgt_eff, 4)),
                                             "dev": round(abs(q - tgt), 3),
                                             "recovery": round(rec(s, native), 4)}) + "\n")
                        if (k + 1) % 200 == 0:
                            fw.flush()
                            print(f"  [{arm}] {k+1}/{n} ...", flush=True)
                    fw.close()
                summ = rebuild_fasta_and_summary(out_path, arm_dir / "seqs.fa",
                                                 arm_dir / "arm_summary.json",
                                                 qnat, native, args.mode, pdb, n,
                                                 per_arm=True)
                if summ is None:
                    print(f"  [{arm}] 未完成({done}/{n})，本轮跳过重建", flush=True)
                    continue
                arm_summaries[arm] = summ
                print(f"  [{arm}] n={summ['n']} mean={summ['mean_charge']:+.2f} "
                      f"dev_of_mean={summ['dev_of_mean']:.2f} "
                      f"per-seq|dev|≤2={summ['per_seq_hit_le2']:.3f} "
                      f"rec={summ['recovery_mean']:.3f}", flush=True)
            summary = {"pdb": pdb, "mode": args.mode,
                       "native_charge": round(qnat, 3), "round_target": t_native,
                       "arms": arm_summaries, "calibration": meta.get("calibration"),
                       "n_per_arm": n, "seed_base_scheme":
                           "SEED_BASE+p_idx*1e6+arm_idx*1e4+k"}
            with open(route_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"  route {args.mode} done → {route_dir/'summary.json'} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # 顶层 meta
    top_meta = {"pH": pH, "temperature": TEMP, "bias_strength": BIAS_STRENGTH,
                "backbone": str(WEIGHTS), "cond_encoder": str(ENC),
                "calibration_table": str(CAL_TABLE), "arms": dict(ARMS),
                "seed_scheme": "seed(p_idx,arm_idx,k)=SEED_BASE+p_idx*1e6+arm_idx*1e4+k",
                "proteins": proteins}
    top_path = out_root / "meta.json"
    if top_path.is_file():
        old = json.load(open(top_path))
        old.update(top_meta)
        top_meta = old
    with open(top_path, "w") as f:
        json.dump(top_meta, f, indent=2, ensure_ascii=False)
    print("\n=== 全部完成 ===", flush=True)


if __name__ == "__main__":
    main()

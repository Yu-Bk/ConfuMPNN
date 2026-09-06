"""exp1/exp2（配体模式）受控对照采样：裸 backbone vs 条件编码器 vs bias-only。

科学问题（compare/plan_exp1_barebackbone_control.md + ablation/plan_exp2_bias_vs_encoder.md）：
    1. 配体模式「条件电荷控制」相对 **裸 LigandMPNN 无条件重设计** 加了多少控制（exp1）。
    2. 去掉 encoder、只用 **简单电荷 logit bias（charge lookahead）** 还剩多少控制（exp2）。

route：
    A  = 裸 backbone 无条件（cond_encoder=None），n_A/蛋白，重设计。
    B  = v14 条件编码器注入，5 臂 native/n2/p2/n8/p8，n_B/臂（直接生成，未校准）。
    C  = 去掉 encoder，仅 guided_sampler + ChargeLookahead logit bias，native/n2/p2 臂，n_C/臂。
    Bcal = （可选，仅某臂直接生成很差时补跑）v14 clean per-protein 校准表推理侧校正。

seed 体系：同蛋白同臂 A/B/C/Bcal 用同一批 seed 偏移（SEED_BASE + 蛋白偏移 + k）。
pH=7.4；净电荷 net_charge(seq,7.4)；LigandMPNN 配体原子上下文 atom_context=25。
温度 0.3（与既往 v14 验证一致）。

用法：
    conda activate confumpnn
    python code/tests/exp_lig_control/run_exp_control_lig.py \
        --device cuda:6 --nA 1000 --nB 1000 --nC 200 \
        --out_dir output/exp_control_lig --log_every 100

输出：
    output/exp_control_lig/
      meta.json                       # 全局元信息
      {pdb}/native.json               # native 序列/电荷
      {pdb}/routeA/sequences.json + seqs.fa
      {pdb}/routeB/{arm}/sequences.json + seqs.fa     (arm in native,n2,p2,n8,p8)
      {pdb}/routeC/{arm}/sequences.json + seqs.fa     (arm in native,n2,p2)
      {pdb}/routeB_cal/{arm}/sequences.json + seqs.fa (仅当 --arms_bcal 指定)
      汇总 json 由 analyze_exp_control_lig.py 生成（summary.json）
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parents[2]      # code/
_ROOT = _CODE_DIR.parent
for p in [str(_CODE_DIR), str(_ROOT / "LigandMPNN")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data_utils import featurize, parse_PDB  # noqa: E402
from src.condition_embedding import make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from src.guided_sampler import GuidedSampler  # noqa: E402
from src.charge_lookahead import make_dynamic_callback  # noqa: E402
from run_guided import (load_model, load_condition_encoder,  # noqa: E402
                        load_calibration, seq_to_string)

# 默认配体模式 backbone + v14 条件编码器（用户指定）
DEFAULT_W = str(_ROOT / "LigandMPNN" / "model_params" / "ligandmpnn_v_32_010_25.pt")
DEFAULT_ENC = str(_ROOT / "output" / "finetune_ligand_v14_rna" / "finetune_epoch050.pt")
# v14 clean per-protein 校准表（仅 routeB_cal 附加组用）
DEFAULT_CAL = str(_ROOT / "output" / "charge_calibration_v14_ligand_clean.json")

# 配体模式 in-10 中选取的好/中/难 3 测试蛋白（依据 2026-09-04 v14 clean 验证 H2/recovery）
PROTEINS = [
    {"pdb": "5O60_E", "path": "data/validation_pdbs/5O60_E.pdb",
     "rank": "好(RNA结合)", "note": "核糖体蛋白E，天然正电；v14 clean H2 5/5 rec 0.39 slope1.78"},
    {"pdb": "1CGE",   "path": "data/validation_pdbs/1CGE.pdb",
     "rank": "中(金属)", "note": "CA+ZN 金属蛋白；v14 clean H2 5/5 rec 0.54 slope0.98"},
    {"pdb": "2FEO",   "path": "data/validation_pdbs/2FEO.pdb",
     "rank": "难(DNA结合)", "note": "DC 核苷酸蛋白；v14 clean H2 0/5 rec 0.32 slope0.93 电荷最弱"},
]

# 电荷臂（Δ 相对 native 电荷，target = round(native_q) + Δ，与既往一致）
ARMS = [("native", 0), ("n2", -2), ("p2", +2), ("n8", -8), ("p8", +8)]
# exp2 规范臂 = native/n2/p2；扩展跑 n8/p8（bias 在极值臂的表现，exp1 对照完整化）。
ARMS_C = [("native", 0), ("n2", -2), ("p2", +2), ("n8", -8), ("p8", +8)]

# Kyte-Doolittle（native identity recovery 之外可选；此处不展开）
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
      "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
      "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
      "Y": -1.3, "V": 4.2, "X": 0.0}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gravy(seq):
    return float(np.mean([KD.get(a, 0.0) for a in seq]))


def identity_rec(seq, native):
    return float(sum(a == b for a, b in zip(seq, native)) / len(native))


def prepare_pdb(pdb_path, pH, num_ligand_atoms=25):
    protein_dict, *_ = parse_PDB(pdb_path, device="cpu")
    L = protein_dict["X"].shape[0]
    native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
    q_nat = float(net_charge(native, pH))
    protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
    fd = featurize(protein_dict, cutoff_for_score=8.0, use_atom_context=True,
                   number_of_ligand_atoms=num_ligand_atoms, model_type="ligand_mpnn")
    fd["batch_size"] = 1
    fd["temperature"] = 0.3
    fd["bias"] = torch.zeros(1, L, 21)
    return fd, L, native, q_nat


def sample_routeA(model, fd, device, pH, n, seed_base):
    """route A：裸 backbone 无条件重设计（enc=None，无 bias）。"""
    seqs, charges, recs, gravs = [], [], [], []
    native = fd["S"][0].cpu().numpy()  # 用于 identity rec 的参考
    native_str = seq_to_string(native)
    for k in range(n):
        torch.manual_seed(seed_base + k)
        fd["randn"] = torch.randn(1, fd["X"].shape[1])
        out = conditioned_sample(model, None, fd,
                                 make_condition_vector(pH, net_charge=0.0), device)
        s = seq_to_string(out["S"][0].cpu().numpy())
        seqs.append(s)
        charges.append(float(net_charge(s, pH)))
        recs.append(identity_rec(s, native_str))
        gravs.append(gravy(s))
    return {"seed_base": seed_base, "n": n, "seqs": seqs, "charges": charges,
            "recs": recs, "gravs": gravs}


def sample_routeB(model, enc, fd, device, pH, arm, n, seed_base, calibrate=False,
                  cal_slope=None, cal_off=None):
    """route B：条件编码器注入。calibrate=True 时用 (tgt-off)/slope 作注入值。"""
    tgt_nominal = arm[0]  # int target (round(qnat)+dq) 已由调用方算好
    tgt_inject = float(tgt_nominal)
    tag = arm[1]
    if calibrate:
        if cal_slope is None:
            raise ValueError("calibrate=True 但无 cal_slope/cal_off")
        tgt_inject = (float(tgt_nominal) - cal_off) / cal_slope
    native_str = seq_to_string(fd["S"][0].cpu().numpy())
    seqs, charges, devs, recs = [], [], [], []
    for k in range(n):
        torch.manual_seed(seed_base + k)
        fd["randn"] = torch.randn(1, fd["X"].shape[1])
        cond_vec = make_condition_vector(pH, net_charge=tgt_inject).to(device)
        out = conditioned_sample(model, enc, fd, cond_vec, device)
        s = seq_to_string(out["S"][0].cpu().numpy())
        q = float(net_charge(s, pH))
        seqs.append(s); charges.append(q)
        devs.append(q - tgt_nominal)   # 名义空间偏差（校准不影响名义 target）
        recs.append(identity_rec(s, native_str))
    return {"seed_base": seed_base, "n": n, "arm": tag, "target": tgt_nominal,
            "target_inject": round(tgt_inject, 3), "calibrate": calibrate,
            "seqs": seqs, "charges": charges, "devs": devs, "recs": recs}


def sample_routeC(model, fd, device, pH, arm, n, seed_base, strength=0.5):
    """route C：去掉 encoder，仅 guided_sampler + charge lookahead logit bias。"""
    tgt_nominal = arm[0]
    native_str = seq_to_string(fd["S"][0].cpu().numpy())
    sampler = GuidedSampler(model, device=device)
    cb = make_dynamic_callback(pH=pH, target_charge=float(tgt_nominal),
                               structure_filter=None, strength=strength)
    seqs, charges, devs, recs = [], [], [], []
    for k in range(n):
        torch.manual_seed(seed_base + k)
        fd["randn"] = torch.randn(1, fd["X"].shape[1])
        out = sampler.sample(fd, bias_callback=cb)
        s = seq_to_string(out["S"][0].cpu().numpy())
        q = float(net_charge(s, pH))
        seqs.append(s); charges.append(q)
        devs.append(q - tgt_nominal)
        recs.append(identity_rec(s, native_str))
    return {"seed_base": seed_base, "n": n, "arm": arm[1], "target": tgt_nominal,
            "strength": strength, "seqs": seqs, "charges": charges, "devs": devs,
            "recs": recs}


def write_seqs(seqs, charges, target, native_seq, q_nat, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "seqs.fa", "w") as f:
        for i, (s, q) in enumerate(zip(seqs, charges)):
            tgt_str = f"{target:+.0f}" if target is not None else "na"
            f.write(f">seed{i} target={tgt_str} charge={q:+.2f}\n{s}\n")
        f.write(f">native charge={q_nat:+.2f}\n{native_seq}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--weights", default=DEFAULT_W)
    ap.add_argument("--cond_encoder", default=DEFAULT_ENC)
    ap.add_argument("--calibration_file", default=DEFAULT_CAL)
    ap.add_argument("--pH", type=float, default=7.4)
    ap.add_argument("--nA", type=int, default=1000, help="route A 每蛋白序列数")
    ap.add_argument("--nB", type=int, default=1000, help="route B 每臂序列数")
    ap.add_argument("--nC", type=int, default=1000, help="route C 每臂序列数")
    ap.add_argument("--strength", type=float, default=0.5, help="route C bias strength")
    ap.add_argument("--out_dir", default=str(_ROOT / "output" / "exp_control_lig"))
    ap.add_argument("--seed_base0", type=int, default=3000)
    ap.add_argument("--run_route", default="A,B,C",
                    help="要跑的 route（逗号分隔：A/B/C）")
    ap.add_argument("--arms_bcal", default="",
                    help="对哪些蛋白（逗号分隔 pdb 名）额外补 routeB_cal 全 5 臂"
                         "（某臂直接生成很差时的现场标定附加组；用 v14 clean per-protein 表）")
    ap.add_argument("--proteins", default="5O60_E,1CGE,2FEO")
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    device = torch.device(args.device)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    run_routes = [x.strip() for x in args.run_route.split(",") if x.strip()]
    sel_pdbs = [p.strip() for p in args.proteins.split(",") if p.strip()]
    bcal_pdbs = [p.strip() for p in args.arms_bcal.split(",") if p.strip()]

    # 加载 backbone 与条件编码器（B/Bcal 用；A/C 不需要 encoder 但也无妨不加载以省显存）
    log(f"加载 backbone: {args.weights}")
    model = load_model(args.weights, device, model_type="auto")
    enc = None
    if "B" in run_routes or any(bcal_pdbs):
        enc = load_condition_encoder(args.cond_encoder, device)

    meta = {"task": "ligand exp1+exp2 control (bare vs conditioned vs bias)",
            "date": time.strftime("%Y-%m-%d"),
            "backbone": args.weights, "cond_encoder": args.cond_encoder,
            "calibration_file": args.calibration_file, "pH": args.pH,
            "temperature": 0.3, "num_ligand_atoms": 25,
            "seed_base0": args.seed_base0, "routeC_strength": args.strength,
            "nA": args.nA, "nB": args.nB, "nC": args.nC,
            "proteins": [], "git": "no-push (main archives)"}

    for pi, pinfo in enumerate(PROTEINS):
        pdb = pinfo["pdb"]
        if pdb not in sel_pdbs:
            continue
        pdb_path = _ROOT / pinfo["path"]
        seed_base = args.seed_base0 + pi * 100000
        log(f"\n===== {pdb} ({pinfo['rank']}) seed_base={seed_base} =====")
        fd, L, native, q_nat = prepare_pdb(str(pdb_path), args.pH)
        tgt_int = int(round(q_nat))
        log(f"  L={L} native_q={q_nat:+.2f} round_target={tgt_int:+d}")

        pout = out_root / pdb
        pout.mkdir(parents=True, exist_ok=True)
        with open(pout / "native.json", "w") as f:
            json.dump({"pdb": pdb, "rank": pinfo["rank"], "note": pinfo["note"],
                       "path": str(pdb_path), "L": L, "native": native,
                       "native_charge": round(q_nat, 3)}, f, indent=2)

        # ---- route A：裸无条件 ----
        if "A" in run_routes:
            adir = pout / "routeA"
            if (adir / "sequences.json").exists():
                log("  route A 已存在，跳过")
            else:
                log(f"  [A] 裸无条件采样 n={args.nA} ...")
                resA = sample_routeA(model, fd, device, args.pH, args.nA, seed_base)
                write_seqs(resA["seqs"], resA["charges"], None, native, q_nat, adir)
                with open(adir / "sequences.json", "w") as f:
                    json.dump({k: resA[k] for k in ("seed_base", "n", "charges",
                                                    "recs", "gravs", "seqs")},
                              f, indent=2)
                log(f"  [A] 完成 mean_q={np.mean(resA['charges']):+.2f} "
                    f"std={np.std(resA['charges']):.2f}")

        # ---- route B：条件编码器，5 臂直接生成 ----
        if "B" in run_routes:
            for arm_tag, dq in ARMS:
                tgt = tgt_int + dq
                adir = pout / "routeB" / f"arm_{arm_tag}"
                if (adir / "sequences.json").exists():
                    log(f"  [B/{arm_tag}] 已存在，跳过")
                    continue
                log(f"  [B/{arm_tag}] target={tgt:+d} 条件生成 n={args.nB} ...")
                t0 = time.time()
                resB = sample_routeB(model, enc, fd, device, args.pH,
                                     (tgt, arm_tag), args.nB, seed_base,
                                     calibrate=False)
                write_seqs(resB["seqs"], resB["charges"], tgt, native, q_nat, adir)
                with open(adir / "sequences.json", "w") as f:
                    json.dump({k: resB[k] for k in
                               ("seed_base", "n", "arm", "target", "target_inject",
                                "calibrate", "charges", "devs", "recs", "seqs")},
                              f, indent=2)
                dev = float(np.mean(resB["charges"])) - tgt
                log(f"  [B/{arm_tag}] 完成 ({time.time()-t0:.0f}s) "
                    f"mean_q={np.mean(resB['charges']):+.2f} dev={dev:+.2f}")

        # ---- route C：bias-only（native/n2/p2）----
        if "C" in run_routes:
            for arm_tag, dq in ARMS_C:
                tgt = tgt_int + dq
                adir = pout / "routeC" / f"arm_{arm_tag}"
                if (adir / "sequences.json").exists():
                    log(f"  [C/{arm_tag}] 已存在，跳过")
                    continue
                log(f"  [C/{arm_tag}] target={tgt:+d} bias-only n={args.nC} ...")
                t0 = time.time()
                resC = sample_routeC(model, fd, device, args.pH, (tgt, arm_tag),
                                     args.nC, seed_base, strength=args.strength)
                write_seqs(resC["seqs"], resC["charges"], tgt, native, q_nat, adir)
                with open(adir / "sequences.json", "w") as f:
                    json.dump({k: resC[k] for k in
                               ("seed_base", "n", "arm", "target", "strength",
                                "charges", "devs", "recs", "seqs")}, f, indent=2)
                dev = float(np.mean(resC["charges"])) - tgt
                log(f"  [C/{arm_tag}] 完成 ({time.time()-t0:.0f}s) "
                    f"mean_q={np.mean(resC['charges']):+.2f} dev={dev:+.2f}")

        # ---- route B_cal：某臂直接生成很差 → 现场标定附加组（per-protein 校准）----
        if pdb in bcal_pdbs:
            cal_slope, cal_off, cal_mode, cal_label = load_calibration(
                args.calibration_file, pdb)
            if cal_slope is None:
                log(f"  !! {pdb} 无 per-protein 校准表项，跳过 routeB_cal")
            else:
                log(f"  routeB_cal 校准: {cal_label} {cal_mode} "
                    f"slope={cal_slope:.3f} off={cal_off:.3f}")
                for arm_tag, dq in ARMS:
                    tgt = tgt_int + dq
                    adir = pout / "routeB_cal" / f"arm_{arm_tag}"
                    if (adir / "sequences.json").exists():
                        continue
                    log(f"  [B_cal/{arm_tag}] target={tgt:+d} n={args.nB} ...")
                    t0 = time.time()
                    res = sample_routeB(model, enc, fd, device, args.pH,
                                        (tgt, arm_tag), args.nB, seed_base,
                                        calibrate=True, cal_slope=cal_slope,
                                        cal_off=cal_off)
                    write_seqs(res["seqs"], res["charges"], tgt, native, q_nat, adir)
                    with open(adir / "sequences.json", "w") as f:
                        json.dump({k: res[k] for k in
                                   ("seed_base", "n", "arm", "target",
                                    "target_inject", "calibrate", "charges",
                                    "devs", "recs", "seqs")}, f, indent=2)
                    log(f"  [B_cal/{arm_tag}] 完成 ({time.time()-t0:.0f}s) "
                        f"mean_q={np.mean(res['charges']):+.2f}")

        meta["proteins"].append({"pdb": pdb, "seed_base": seed_base, "L": L,
                                 "native_charge": round(q_nat, 3)})
        with open(out_root / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    log("\n=== 采样全部完成 ===")


if __name__ == "__main__":
    main()

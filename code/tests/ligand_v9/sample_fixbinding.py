"""Task2（2026-09-04）：配体结合残基固定为 native 后的条件生成。

科学问题：v14 配体（RNA/DNA + A1 global）条件生成的"删带电残基捷径"（删减偏
结合口袋）在把结合残基固定为 native 后减弱多少？电荷命中(H2)/电荷聚集(H3)代价？

方法（与 validate_generalization.py 同源，但**独立脚本**，不动 Task1/3 的
validate_generalization 接口）：
  - binding 残基 = Cα 距配体重原子 ≤ cutoff(8Å) 的蛋白残基（pocket_residues 口径，
    与 v12/v14 监督一致；配体重原子来自 parse_PDB 的 Y：HETATM/核酸，非水非氢）。
  - fix = 把 protein_dict["chain_mask"] 在 binding 位置置 0 → 共享采样核心
    guided_sampler.guided_sample（conditioned_sampler 调用链）原生支持：
    自回归解码时 chain_mask=0 的位置强制放 native（S_t = native），位置按
    argsort((chain_mask+0.0001)*|randn|) 排在最前，后续位置可正常 attend。
    无需改动共享采样核心（conditioned_sampler/guided_sampler）。
  - 采样 in-10 × 5 臂(native/n2/p2/n8/p8) × n=40，seed_base=2000，per-protein 校准
    （charge_calibration_v14_ligand_clean.json，10 蛋白全在表内；表外回退 global）。
  - 只测电荷(H2)+组成(删减)+聚集(H3)，不做 ESMFold/Tm/Sol。

输出结构（对齐 unfix 基线 generalization_ligand_v14_clean 以复用下游）：
  {out_dir}/ligand/<pdb>/pH7.4/arm_<arm>/seqs.fa   # n 条生成 + 末尾 native 参考行
  {out_dir}/ligand/<pdb>/validation.json
  {out_dir}/fixed/<pdb>_fixed.json                 # binding 残基 mask/明细

用法（项目根，confumpnn 环境）：
  PYTHONPATH=code python code/tests/ligand_v9/sample_fixbinding.py \
      --manifest data/validation_pdbs/validation_manifest_v14_in.json \
      --out_dir output/fixbinding_v14 \
      --cond_encoder output/finetune_ligand_v14_rna/finetune_epoch050.pt \
      --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
      --n 40 --device cuda:6 --pH 7.4 \
      --calibrate auto --calibration_file output/charge_calibration_v14_ligand_clean.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_CODE_DIR.parent / "LigandMPNN"))

from data_utils import featurize, parse_PDB  # noqa: E402
from src.condition_embedding import make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from run_guided import (load_model, load_condition_encoder, seq_to_string,
                        load_calibration)  # noqa: E402

KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
      "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
      "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
      "Y": -1.3, "V": 4.2, "X": 0.0}
ARMS = [("native", 0), ("n2", -2), ("p2", +2), ("n8", -8), ("p8", +8)]


def gravy(seq):
    return np.mean([KD.get(a, 0.0) for a in seq])


def binding_residues(protein_dict, cutoff=8.0):
    """结合残基 = 与配体原子（Y，HETATM/核酸，非水非氢）距离 < cutoff 的残基索引（Cα）。"""
    Y = protein_dict.get("Y")
    X = protein_dict["X"]
    if Y is None or Y.numel() == 0:
        return None
    Yc = Y.reshape(-1, 3).cpu().numpy()
    CA = X[:, 1, :].cpu().numpy()
    if len(Yc) == 0:
        return None
    d = np.linalg.norm(CA[:, None, :] - Yc[None, :, :], axis=-1)
    return np.where(d.min(axis=1) < cutoff)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", default="output/fixbinding_v14")
    ap.add_argument("--cond_encoder", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--pH", type=float, default=7.4)
    ap.add_argument("--seed_base", type=int, default=2000)
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--num_ligand_atoms", type=int, default=25)
    ap.add_argument("--cutoff", type=float, default=8.0,
                    help="结合残基半径（Cα-配体重原子，与 pocket 口径一致）")
    ap.add_argument("--arms", default="native,n2,p2,n8,p8")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--calibrate", default="auto", choices=["auto", "global", "off"])
    ap.add_argument("--calibration_file",
                    default="output/charge_calibration_v14_ligand_clean.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    arm_map = dict(ARMS)
    sel_arms = [a for a in args.arms.split(",") if a in arm_map]

    enc = load_condition_encoder(args.cond_encoder, device)
    model = load_model(args.weights, device, model_type="ligand_mpnn")

    manifest = json.load(open(args.manifest))
    items = manifest["items"][args.start: args.end]
    print(f"Task2 fix-binding: 处理 {len(items)} 蛋白，臂={sel_arms}，n={args.n}，"
          f"pH={args.pH}，cutoff={args.cutoff}Å", flush=True)

    out_root = Path(args.out_dir)
    fixed_root = out_root / "fixed"
    fixed_root.mkdir(parents=True, exist_ok=True)
    feats = dict(model_type="ligand_mpnn", use_atom_context=True,
                 number_of_ligand_atoms=args.num_ligand_atoms)

    for it in items:
        pdb = it["pdb"]
        pdb_path = Path(it["path"])
        protein_dict, _, _, icodes, _ = parse_PDB(str(pdb_path))
        L = protein_dict["X"].shape[0]
        native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
        q_nat = float(net_charge(native, args.pH))

        bind = binding_residues(protein_dict, cutoff=args.cutoff)
        if bind is None or len(bind) == 0:
            print(f"  !! {pdb} 无配体原子，无法定义结合残基，跳过", flush=True)
            continue
        chain_letters = list(protein_dict["chain_letters"])
        R_idx = list(protein_dict["R_idx"].cpu().numpy())
        fixed_names = []
        for i in sorted(int(b) for b in bind):
            code = str(chain_letters[i]) + str(R_idx[i]) + (
                str(icodes[i]) if icodes is not None else "")
            fixed_names.append({"index": int(i), "chain": str(chain_letters[i]),
                                "resnum": int(R_idx[i]), "name": code})
        chain_mask = np.ones(L, dtype=np.int32)
        chain_mask[bind] = 0
        n_fixed = int((chain_mask == 0).sum())
        frac_fixed = n_fixed / L
        print(f"\n=== {pdb} cat={it.get('cat')} L={L} native_charge@7.4={q_nat:+.2f} "
              f"binding={n_fixed} ({frac_fixed:.1%}) ===", flush=True)
        # 存 binding 明细
        fixed_meta = {"pdb": pdb, "L": L, "cutoff": args.cutoff,
                      "native_charge": round(q_nat, 2),
                      "n_fixed": n_fixed, "frac_fixed": round(frac_fixed, 4),
                      "binding_residues": fixed_names}
        with open(fixed_root / f"{pdb}_fixed.json", "w") as f:
            json.dump(fixed_meta, f, indent=2, ensure_ascii=False)

        protein_dict["chain_mask"] = torch.tensor(chain_mask, dtype=torch.int32)
        fd = featurize(protein_dict, cutoff_for_score=8.0, **feats)
        fd["batch_size"] = 1
        fd["temperature"] = 0.3
        fd["bias"] = torch.zeros(1, L, 21)

        cal_slope = cal_off = None
        if args.calibrate != "off":
            cal_slope, cal_off, cal_mode, _ = load_calibration(
                args.calibration_file, pdb, force_global=(args.calibrate == "global"))
            print(f"    校准: {cal_mode} (slope={cal_slope:.3f} intercept={cal_off:.3f})", flush=True)

        summary = {"pdb": pdb, "cat": it.get("cat"), "L": L, "mode": "ligand",
                   "native": native, "native_charge": round(q_nat, 2),
                   "fix": {"cutoff": args.cutoff, "n_fixed": n_fixed,
                           "frac_fixed": round(frac_fixed, 4)},
                   "arms": {}}
        for arm in sel_arms:
            dq = arm_map[arm]
            tgt = int(round(q_nat)) + dq
            tgt_eff = tgt
            if cal_slope is not None:
                tgt_eff = (tgt - cal_off) / cal_slope
            charges, recs, pkt_recs, gravs, seqs, fix_mm = [], [], [], [], [], []
            for k in range(args.n):
                torch.manual_seed(args.seed_base + k)
                fd["randn"] = torch.randn(1, L)
                cond_vec = make_condition_vector(args.pH, net_charge=tgt_eff)
                out = conditioned_sample(model, enc, fd, cond_vec, device)
                seq = seq_to_string(out["S"][0].cpu().numpy())
                seqs.append(seq)
                charges.append(float(net_charge(seq, args.pH)))
                recs.append(sum(a == b for a, b in zip(seq, native)) / L)
                # 固定位必须 100% native
                mm = sum(1 for i in bind if seq[i] != native[i])
                fix_mm.append(mm)
                pkt_recs.append(1.0 if len(bind) else None)  # 固定 → 口袋恒 native
                gravs.append(gravy(seq))
            mean_c = float(np.mean(charges))
            dev = abs(mean_c - tgt)
            n_fix_mm = int(sum(fix_mm))
            arm_dir = out_root / "ligand" / pdb / f"pH{args.pH}" / f"arm_{arm}"
            arm_dir.mkdir(parents=True, exist_ok=True)
            fa = arm_dir / "seqs.fa"
            with open(fa, "w") as f:
                for i, (s, c) in enumerate(zip(seqs, charges)):
                    f.write(f">seed_{args.seed_base+i} arm={arm} target={tgt:+.0f} "
                            f"charge={c:+.2f}\n{s}\n")
                f.write(f">native charge={q_nat:+.2f}\n{native}\n")
            summary["arms"][arm] = {
                "target": tgt, "mean_charge": round(mean_c, 2),
                "std_charge": round(float(np.std(charges)), 2),
                "dev": round(dev, 2),
                "recovery": round(float(np.mean(recs)), 3),
                "pocket_recovery": 1.0,
                "gravy_mean": round(float(np.mean(gravs)), 3),
                "n_generated": len(seqs),
                "fixed_mismatch_total": n_fix_mm,
            }
            print(f"  [{arm}] target={tgt:>4} mean={mean_c:+6.2f} dev={dev:.2f} "
                  f"rec={np.mean(recs):.3f} fixed_mm={n_fix_mm}", flush=True)
        with open(out_root / "ligand" / pdb / "validation.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  已写 {pdb} 结果", flush=True)

    print("\n=== Task2 fix-binding 采样完成 ===", flush=True)


if __name__ == "__main__":
    main()

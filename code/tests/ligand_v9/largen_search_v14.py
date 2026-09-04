"""Task3 大样本"三达标"搜索 (v14 配体 in-10, GPU4)。

科学问题：把每臂采样数放大到 n=200，是否存在**同时满足**
  ① 电荷达标  |net_charge(seq) − target| ≤ 2
  ② 不重删带电残基  D/E+K/R ≥ 0.7×native（删除倍率 ≥ 0.7）
  ③ 无电荷聚集 H3 合法（结构感知过滤 4 规则违规率 ≤ native_ref + 0.05，与 H3 判据同口径）
的序列？——判定 v14 的删减/电荷失败是"模型从不生成合格序列"还是"稀有事件、多采样可救"。

口径一致性（与 validate_generalization.py 对齐）：
  - 每蛋白 5 臂 native/n2/p2/n8/p8，Δ = 0/−2/+2/−8/+8 相对 round(native_charge)
  - per-protein 电荷校准：tgt_eff = (tgt − intercept) / slope
    （charge_calibration_v14_ligand_clean.json；缺失蛋白回退 global）
  - 条件注入采样：conditioned_sample(model, enc, fd, cond_vec)，temperature=0.3，
    seed_base=2000+k（k=0..n-1），fd randn 同 validate——前 50 条可复现参考 n50。
  - ligand_mpnn backbone（atom25），use_atom_context=True，num_ligand_atoms=25。

H3 口径（与 h3_charge_legality.py 逐序列一致）：
  - coords = ref 骨架 Cα（output/generalization_ligand_v14_clean/ref/<pdb>_ref.pdb）
  - 4 规则 full union count = count_violations(coords, seq_int, pos, neg, cfg)
  - native_ref = 同骨架 native 序列的 full count
  - 每序列 H3 合法判定：full_count/L ≤ native_ref/L + 0.05（等价 H3 arm PASS 的 per-seq 化）
  - 额外记录 local 规则计数（charge_cluster+salt_bridge+core_charge，去掉 R4 巨型连通分量退化）

输出：
  output/largen_v14/<pdb>_arm_<arm>/seqs.fa + stats.json
  output/largen_v14/summary.json

断点续跑：每 arm 采样先读已有 seqs.fa（快照），只补缺失 seed；每 10 条原子化重写快照。

用法（项目根）：
  PYTHONPATH=code python code/tests/ligand_v9/largen_search_v14.py \
    --manifest data/validation_pdbs/validation_manifest_v14_in.json \
    --out_dir output/largen_v14 \
    --ref_dir output/generalization_ligand_v14_clean/ref \
    --cond_encoder output/finetune_ligand_v14_rna/finetune_epoch050.pt \
    --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
    --calibration_file output/charge_calibration_v14_ligand_clean.json \
    --n 200 --device cuda:4
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
_CODE_DIR = ROOT / "code"
for p in (str(_CODE_DIR), str(ROOT / "LigandMPNN"), str(_CODE_DIR / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_utils import featurize, parse_PDB  # noqa: E402
from src.condition_embedding import make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from src.structure_aware_filter import default_config, pH_adaptive_charged_aa  # noqa: E402
from run_guided import (load_calibration, load_condition_encoder,  # noqa: E402
                        load_model, seq_to_string)
import h3_charge_legality as h3  # noqa: E402

ARMS = [("native", 0), ("n2", -2), ("p2", +2), ("n8", -8), ("p8", +8)]

# 三达标阈值
DEV_MAX = 2.0            # |q - target| ≤ 2
DEL_MIN = 0.7            # D/E+K/R ≥ 0.7×native
H3_SLACK = 0.05          # H3 违规率 ≤ native_ref + 5pp


def count_dk(seq):
    de = sum(1 for a in seq if a in "DE")
    kr = sum(1 for a in seq if a in "KR")
    return de + kr, de, kr


def parse_fa(fa_path, native_len):
    """读 fasta → 记录（seed, seq）；只保留长度 == native_len 的完整行。"""
    recs = []
    name = None
    buf = []
    for line in open(fa_path):
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                seq = "".join(buf)
                if len(seq) == native_len and not name.startswith("native"):
                    recs.append((name, seq))
            name = line[1:]
            buf = []
        elif line:
            buf.append(line)
    if name is not None:
        seq = "".join(buf)
        if len(seq) == native_len and not name.startswith("native"):
            recs.append((name, seq))
    out = {}
    for nm, seq in recs:
        seed = None
        for tok in nm.split():
            if tok.startswith("seed_"):
                try:
                    seed = int(tok.split("_")[1])
                except (IndexError, ValueError):
                    seed = None
        if seed is not None:
            out[seed] = seq
    return out


def write_fa(fa_path, seed_seq, native):
    """原子写 fasta：seed_seq dict[int,str] 按 k 升序 + native 行。"""
    tmp = fa_path.with_suffix(".fa.tmp")
    with open(tmp, "w") as f:
        for k in sorted(seed_seq):
            f.write(f">seed_{k}\n{seed_seq[k]}\n")
        f.write(f">native\n{native}\n")
    os.replace(tmp, fa_path)


def sample_arm(model, enc, fd, L, pH, tgt, tgt_eff, n, seed_base, device, fa_path,
               native, arm_dir, snap=10):
    """补采缺失 seed 至 n；返回有序 seq 列表。"""
    done = parse_fa(fa_path, L) if fa_path.exists() else {}
    todo = [k for k in range(n) if (seed_base + k) not in done]
    if not todo:
        return [done[seed_base + k] for k in range(n)]
    print(f"    arm {fa_path.parent.name}: 已有 {len(done)} 条, 需补采 {len(todo)} 条", flush=True)
    t0 = time.time()
    cnt = 0
    for k in todo:
        seed = seed_base + k
        torch.manual_seed(seed)
        fd["randn"] = torch.randn(1, L)
        cond_vec = make_condition_vector(pH, net_charge=tgt_eff)
        out = conditioned_sample(model, enc, fd, cond_vec, device)
        seq = seq_to_string(out["S"][0].cpu().numpy())
        done[seed] = seq
        cnt += 1
        if cnt % snap == 0 or cnt == len(todo):
            write_fa(fa_path, done, native)
            dt = time.time() - t0
            print(f"      ... {cnt}/{len(todo)} ({dt/cnt:.2f}s/seq, "
                  f"target={tgt})", flush=True)
    write_fa(fa_path, done, native)
    return [done[seed_base + k] for k in range(n)]


def h3_metrics(coords, seq, pH, native_rate):
    """逐序列 H3 指标。返回 dict。"""
    pos_aa, neg_aa = pH_adaptive_charged_aa(pH)
    seq_int = h3.seq_to_int(seq)
    L = len(seq)
    cfg = default_config()
    full, _, per = h3.count_violations(coords, seq_int, pos_aa, neg_aa, cfg)
    # local = R1+R2+R3（把 R4 阈值调超高 → 永不触发，得到 union）
    cfg2 = default_config()
    cfg2["same_sign_cluster"]["threshold"] = L + 1
    local, _, per2 = h3.count_violations(coords, seq_int, pos_aa, neg_aa, cfg2)
    rate = full / L
    return {
        "full_viol": int(full), "rate": round(float(rate), 4),
        "h3_pass": bool(rate <= native_rate + H3_SLACK),
        "local_viol": int(local),
        "rules": {k: int(v) for k, v in per.items()},
    }


def pareto_front(points):
    """点 (dev, del)；dominated 按 dev 越小 / del 越大越好。返回 Pareto 索引列表。"""
    pts = [(d, dd, i) for i, (d, dd) in enumerate(points)]
    non_dom = []
    for a in pts:
        dominated = False
        for b in pts:
            if a is b:
                continue
            if b[0] <= a[0] and b[1] >= a[1] and (b[0] < a[0] or b[1] > a[1]):
                dominated = True
                break
        if not dominated:
            non_dom.append(a)
    non_dom.sort(key=lambda x: x[0] + max(0.0, 1.0 - x[1]))
    return [a[2] for a in non_dom]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", default="output/largen_v14")
    ap.add_argument("--ref_dir", default="output/generalization_ligand_v14_clean/ref")
    ap.add_argument("--cond_encoder", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--calibration_file", default="output/charge_calibration_v14_ligand_clean.json")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--pH", type=float, default=7.4)
    ap.add_argument("--seed_base", type=int, default=2000)
    ap.add_argument("--device", default="cuda:4")
    ap.add_argument("--num_ligand_atoms", type=int, default=25)
    ap.add_argument("--arms", default="native,n2,p2,n8,p8")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--snap", type=int, default=10)
    args = ap.parse_args()

    device = torch.device(args.device)
    arm_map = dict(ARMS)
    sel_arms = [a for a in args.arms.split(",") if a in arm_map]
    assert args.device.startswith("cuda:4"), "本任务只允许 cuda:4"

    manifest = json.load(open(args.manifest))
    items = manifest["items"][args.start:args.end]

    out_root = Path(args.out_dir)
    ref_root = Path(args.ref_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"加载模型: encoder={Path(args.cond_encoder).name} backbone={Path(args.weights).name} "
          f"device={args.device}", flush=True)
    enc = load_condition_encoder(args.cond_encoder, device)
    model = load_model(args.weights, device, model_type="auto")

    summary = {"label": "v14_largen_triple_search", "n": args.n, "pH": args.pH,
               "criteria": {"dev_max": DEV_MAX, "del_min": DEL_MIN,
                            "h3_slack": H3_SLACK},
               "calibration_file": args.calibration_file, "proteins": {}}

    for it in items:
        pdb = it["pdb"]
        pdb_path = Path(it["path"])
        ref_pdb = ref_root / f"{pdb}_ref.pdb"
        if not ref_pdb.exists():
            print(f"!! {pdb} 缺 ref {ref_pdb}，跳过", flush=True)
            continue

        protein_dict = parse_PDB(str(pdb_path))[0]
        protein_dict["chain_mask"] = torch.ones(protein_dict["X"].shape[0],
                                                dtype=torch.int32)
        L = protein_dict["X"].shape[0]
        native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
        q_nat = float(net_charge(native, args.pH))
        nat_dk, nat_de, nat_kr = count_dk(native)
        coords = h3.ca_coords_from_pdb(str(ref_pdb))
        assert coords.shape[0] == L, f"{pdb} coords {coords.shape} != L {L}"
        nat_h3 = h3_metrics(coords, native, args.pH, 0.0)  # native_rate=0 只取 count
        native_h3_full = nat_h3["full_viol"]
        native_rate = native_h3_full / L

        cal_slope, cal_off, cal_mode, _ = load_calibration(args.calibration_file, pdb)
        print(f"\n=== {pdb} cat={it.get('cat')} L={L} q_nat={q_nat:+.2f} "
              f"DK_nat={nat_dk} (DE={nat_de}/KR={nat_kr}) H3_native={native_h3_full} "
              f"({native_rate:.3f}) cal={cal_mode or 'NONE'}", flush=True)

        feats = dict(model_type="ligand_mpnn", use_atom_context=True,
                     number_of_ligand_atoms=args.num_ligand_atoms)
        fd = featurize(protein_dict, cutoff_for_score=8.0, **feats)
        fd["batch_size"] = 1
        fd["temperature"] = 0.3
        fd["bias"] = torch.zeros(1, L, 21)

        prot_stats = {"pdb": pdb, "cat": it.get("cat"), "L": L, "pH": args.pH,
                      "native": native, "native_charge": round(q_nat, 3),
                      "native_dk": nat_dk, "native_de": nat_de, "native_kr": nat_kr,
                      "native_h3_full_viol": native_h3_full,
                      "native_h3_rate": round(native_rate, 4),
                      "calibration": {"mode": cal_mode,
                                      "slope": cal_slope, "intercept": cal_off},
                      "arms": {}}

        for arm in sel_arms:
            dq = arm_map[arm]
            tgt = int(round(q_nat)) + dq
            tgt_eff = (tgt - cal_off) / cal_slope if cal_slope else float(tgt)
            arm_dir = out_root / f"{pdb}_arm_{arm}"
            arm_dir.mkdir(parents=True, exist_ok=True)
            fa = arm_dir / "seqs.fa"

            seqs = sample_arm(model, enc, fd, L, args.pH, tgt, tgt_eff, args.n,
                              args.seed_base, device, fa, native,
                              arm_dir, snap=args.snap)
            n = len(seqs)

            rows = []
            for k, seq in enumerate(seqs):
                q = float(net_charge(seq, args.pH))
                dev = abs(q - tgt)
                dk, de, kr = count_dk(seq)
                del_ratio = dk / nat_dk if nat_dk else float("inf")
                hh = h3_metrics(coords, seq, args.pH, native_rate)
                chg_ok = dev <= DEV_MAX
                del_ok = del_ratio >= DEL_MIN
                triple = bool(chg_ok and del_ok and hh["h3_pass"])
                rows.append({"k": k, "seq": seq, "charge": round(q, 3),
                             "dev": round(dev, 3), "dk": dk, "de": de, "kr": kr,
                             "del_ratio": round(del_ratio, 4),
                             "chg_ok": chg_ok, "del_ok": del_ok,
                             "h3_pass": hh["h3_pass"], "h3_full": hh["full_viol"],
                             "h3_local": hh["local_viol"], "triple": triple})

            devs = np.array([r["dev"] for r in rows])
            dels = np.array([r["del_ratio"] for r in rows])
            h3fs = np.array([r["h3_full"] for r in rows])
            h3ls = np.array([r["h3_local"] for r in rows])
            n_chg = int(sum(r["chg_ok"] for r in rows))
            n_del = int(sum(r["del_ok"] for r in rows))
            n_h3 = int(sum(r["h3_pass"] for r in rows))
            n_triple = int(sum(r["triple"] for r in rows))

            # 前缀存在率（前 10/25/50/100/200）
            prefix = {}
            for pn in (10, 25, 50, 100, 200):
                if n >= pn:
                    prefix[f"p{pn}"] = int(sum(r["triple"] for r in rows[:pn]))
                else:
                    prefix[f"p{pn}"] = None

            # Pareto(dev, del) 前沿示例
            pfront = pareto_front([(r["dev"], r["del_ratio"]) for r in rows])
            pareto_ex = []
            for idx in pfront[:8]:
                r = rows[idx]
                pareto_ex.append({"k": r["k"], "dev": r["dev"], "del_ratio": r["del_ratio"],
                                  "triple": r["triple"], "seq": r["seq"]})

            # 主因分解（仅看未三达标序列会卡在哪）
            fails = {"charge": 0, "deletion": 0, "h3": 0}
            # 定义"主因"：对所有序列，统计各自不满足的判据次数；再降序
            fail_charge = n - n_chg
            fail_del = n - n_del
            fail_h3 = n - n_h3
            main_cause = None
            if n_triple < n:
                order = sorted([("charge", fail_charge), ("deletion", fail_del),
                                ("h3", fail_h3)], key=lambda x: -x[1])
                main_cause = [c for c, f in order if f == order[0][1]]

            n_fail_only = {}
            for r in rows:
                if r["triple"]:
                    continue
                key = (("C" if not r["chg_ok"] else "") +
                       ("D" if not r["del_ok"] else "") +
                       ("H" if not r["h3_pass"] else "")) or "?"
                n_fail_only[key] = n_fail_only.get(key, 0) + 1

            arm_stats = {
                "arm": arm, "target": tgt, "tgt_eff": round(tgt_eff, 3),
                "n": n,
                "charge": {"pass": n_chg, "fail": fail_charge,
                           "dev_min": round(float(devs.min()), 3),
                           "dev_mean": round(float(devs.mean()), 3),
                           "dev_max": round(float(devs.max()), 3)},
                "deletion": {"pass": n_del, "fail": fail_del,
                             "ratio_min": round(float(dels.min()), 4),
                             "ratio_mean": round(float(dels.mean()), 4),
                             "ratio_max": round(float(dels.max()), 4)},
                "h3": {"pass": n_h3, "fail": fail_h3,
                       "full_viol_mean": round(float(h3fs.mean()), 2),
                       "local_viol_mean": round(float(h3ls.mean()), 2),
                       "native_rate": round(native_rate, 4)},
                "triple_pass": n_triple,
                "triple_rate": round(n_triple / n, 4) if n else None,
                "prefix": prefix,
                "main_cause_if_none": main_cause,
                "fail_breakdown": n_fail_only,
                "pareto_examples": pareto_ex,
                "best": None,
            }
            # 最佳三达标示例
            tri = [r for r in rows if r["triple"]]
            if tri:
                best = min(tri, key=lambda r: (r["dev"], -r["del_ratio"]))
                arm_stats["best"] = {k: best[k] for k in
                                     ("k", "seq", "charge", "dev", "dk", "de",
                                      "kr", "del_ratio", "h3_full", "h3_local")}

            with open(arm_dir / "stats.json", "w") as f:
                json.dump({**arm_stats,
                           "seq_rows": [{kk: r[kk] for kk in
                                        ("k", "dev", "del_ratio", "charge",
                                         "chg_ok", "del_ok", "h3_pass",
                                         "h3_full", "h3_local", "triple")}
                                       for r in rows]},
                          f, indent=2)
            prot_stats["arms"][arm] = arm_stats
            print(f"  [{arm}] target={tgt:>4} eff={tgt_eff:+.2f} n={n} "
                  f"pass C={n_chg} D={n_del} H={n_h3} 三达标={n_triple} "
                  f"({n_triple/n:.3f}) dev_m={devs.mean():.2f} del_m={dels.mean():.3f} "
                  f"主因={main_cause}", flush=True)

        with open(out_root / f"{pdb}_summary.json", "w") as f:
            json.dump(prot_stats, f, indent=2, ensure_ascii=False)
        summary["proteins"][pdb] = {k: prot_stats[k] for k in
                                    ("pdb", "cat", "L", "native_charge", "native_dk",
                                     "native_h3_rate", "arms")}

    # 全量汇总：也直接引用每 arm stats 里已算好的字段（summary 已含 arms）
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n=== 完成 ===", flush=True)


if __name__ == "__main__":
    main()

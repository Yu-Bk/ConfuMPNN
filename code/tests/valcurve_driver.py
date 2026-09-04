"""逐 epoch 验证拟合曲线驱动（轨 A：蛋白 v12.2/v12.3，2026-09-04）。

在"最终真未见验证集"（CATH 15% hold-out 1176 ± 长蛋白/深负补充域）上，
对某一模型版本的一串 epoch checkpoint，用 validate_holdout.py 同款逻辑
（parse_PDB → featurize(protein) → conditioned_sample → net_charge → dev/H2/recovery）
测每 epoch 的 H2(dev≤2 命中率)/mean_dev/recovery，输出 {epoch: stats} 曲线。

设计要点：
  1. 域子集抽样一次（seed 固定），各 epoch 在同一批域上评估 → 干净的拟合曲线。
  2. 结构 parse/featurize 与 epoch 无关 → 只做一次，各 epoch 复用（省算力）。
  3. 抽样分层：base(hold-out) n_base + supplement n_supp（若给 --supp_labels，
     supplement 域几乎全抽以覆盖长蛋白/深负区；不给则仅 base 均匀抽 n_dom）。
  4. 校准：默认不校准（开环响应，所有 epoch/版本一致口径，看模型响应随 epoch 收敛）；
     也可 --calibration_file + --force_global 复现 validate_holdout 的全局校准口径。
  5. ⚠️ 必须 GPU（conditioned_sample 前向）；本脚本只负责驱动，等 GPU6 空后由
     主会话调度运行。

用法（项目根，GPU 空时）：
  PYTHONPATH=code /home/baokun_yu/miniconda3/envs/confumpnn/bin/python code/tests/valcurve_driver.py \
     --tag v12_2 --ckpt_dir output/finetune_v12_2 --start_epoch 1 --end_epoch 30 --epoch_step 2 \
     --labels data/cath/labels_holdout_train.npz --dompdb data/cath/S40/dompdb_pdb \
     --supp_labels data/cath/labels_v12_3_valsupp_a.npz --supp_dompdb data/cath/S40/dompdb_pdb \
     --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt \
     --n_base 18 --n_supp 7 --n_per 5 --seed 42 --device cuda:6 \
     --out output/valcurve_v12_2.json

  （v12.3 把 --tag v12_3 --ckpt_dir output/finetune_v12_3 --end_epoch 40 同上）

补充 PDB 说明：15 个补充域中 12 个放在 data/cath/S40/dompdb_valsupp/（从 dompdb 拷贝，
  供 parse_PDB），3 个已在 dompdb_pdb/。运行 track A valcurve 时 base(hold-out 1176) 全在
  dompdb_pdb/，故 --dompdb 指 dompdb_pdb、--supp_dompdb 指 dompdb_valsupp。
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
AP.add_argument("--tag", required=True, help="输出标识（如 v12_2 / v12_3）")
AP.add_argument("--ckpt_dir", required=True, help="epoch checkpoint 目录（含 finetune_epochNNN.pt）")
AP.add_argument("--start_epoch", type=int, default=1)
AP.add_argument("--end_epoch", type=int, default=30)
AP.add_argument("--epoch_step", type=int, default=2, help="每几个 epoch 取一点（省算力）")
AP.add_argument("--labels", required=True, help="base 验证集 npz（如 hold-out 1176）")
AP.add_argument("--dompdb", required=True, help="base 域 PDB 目录")
AP.add_argument("--supp_labels", default=None, help="补充验证集 npz（可选）")
AP.add_argument("--supp_dompdb", default=None, help="补充域 PDB 目录（可选）")
AP.add_argument("--n_base", type=int, default=18, help="每 epoch 从 base 抽样的域数")
AP.add_argument("--n_supp", type=int, default=7, help="每 epoch 从补充域抽样的域数（≤补充域总数）")
AP.add_argument("--mode", default="protein", choices=["protein", "ligand"],
                help="protein=MoMPNN 蛋白模式（v12.2/v12.3）；ligand=LigandMPNN 配体模式（v14）")
AP.add_argument("--num_ligand_atoms", type=int, default=25)
AP.add_argument("--n_per", type=int, default=5, help="每 (域,pH) 采样条数")
AP.add_argument("--seed", type=int, default=42)
AP.add_argument("--temperature", type=float, default=0.3)
AP.add_argument("--weights", default=str(_PROJECT_DIR / "MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt"))
AP.add_argument("--calibration_file", default=None, help="给定时用全局校准（force_global）")
AP.add_argument("--device", default="cuda:6")
AP.add_argument("--out", required=True)
ARGS = AP.parse_args()


def _find_pdb(did, dompdb, dompdb2=None):
    """先在主 dompdb 找 PDB 文件，找不到再到补充目录找。

    兼容两种 domain_id 风格：蛋白 CATH 域 '1d8wC00'（→ {did}.pdb）与
    配体 labels 的 '4V4T_AB.pdb'（did 自带 .pdb 后缀）。
    """
    fname = did if did.endswith(".pdb") else f"{did}.pdb"
    for d in (dompdb, dompdb2):
        if not d:
            continue
        p = Path(d) / fname
        if p.is_file():
            return p
    return None


def load_npz_labels(path):
    d = np.load(path, allow_pickle=True)
    n_dom = len(d["domain_ids"])
    n_pH = d["pH"].shape[0] // n_dom
    return {
        "domain_ids": [str(x) for x in d["domain_ids"]],
        "pH": np.asarray(d["pH"]).reshape(n_dom, n_pH),
        "charge": np.asarray(d["charge"]).reshape(n_dom, n_pH),
    }


def main():
    device = torch.device(ARGS.device)
    base = load_npz_labels(ARGS.labels)
    supp = load_npz_labels(ARGS.supp_labels) if ARGS.supp_labels else None

    # 每 epoch 同一批域：固定 seed 抽 domain index
    rng = random.Random(ARGS.seed)

    # 抽样（先 base 后 supp）
    pick_base = sorted(rng.sample(range(len(base["domain_ids"])), ARGS.n_base))
    dom_infos = [("base", i) for i in pick_base]
    n_supp_actual = 0
    if supp is not None:
        n_avail = len(supp["domain_ids"])
        n_supp_actual = min(ARGS.n_supp, n_avail)
        pick_supp = sorted(rng.sample(range(n_avail), n_supp_actual))
        dom_infos += [("supp", i) for i in pick_supp]
    print(f"抽样域：base={len(pick_base)} supp={n_supp_actual} 合计={len(dom_infos)}", flush=True)

    # 预解析 + 预特征化（与 epoch 无关，只做一次）
    feats_list = []  # [(did, L, native, fd, targets)]
    n_parse_ok = 0
    for setname, idx in dom_infos:
        src = base if setname == "base" else supp
        did = src["domain_ids"][idx]
        p = _find_pdb(did, ARGS.dompdb, ARGS.supp_dompdb)
        if p is None:
            print(f"  !! {did} 找不到 PDB，跳过", flush=True)
            continue
        try:
            protein_dict, *_ = parse_PDB(str(p), device="cpu", parse_all_atoms=False)
        except Exception as e:
            print(f"  !! {did} parse_PDB 失败跳过: {e}", flush=True)
            continue
        L = protein_dict["X"].shape[0]
        native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
        if len(native) != L:
            print(f"  !! {did} seq/L 不一致，跳过", flush=True)
            continue
        protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
        if ARGS.mode == "protein":
            feats = dict(model_type="protein_mpnn", use_atom_context=False,
                         number_of_ligand_atoms=0)
        else:
            feats = dict(model_type="ligand_mpnn", use_atom_context=True,
                         number_of_ligand_atoms=ARGS.num_ligand_atoms)
        fd = featurize(protein_dict, cutoff_for_score=8.0, **feats)
        fd["batch_size"] = 1
        fd["temperature"] = ARGS.temperature
        fd["bias"] = torch.zeros(1, L, 21)
        # 该域 8 个 (pH, target)
        arms = []
        for j in range(src["pH"].shape[1]):
            arms.append((float(src["pH"][idx, j]), round(float(src["charge"][idx, j]))))
        feats_list.append({"did": did, "L": L, "native": native, "fd": fd, "arms": arms})
        n_parse_ok += 1
    print(f"parse 成功 {n_parse_ok}/{len(dom_infos)}", flush=True)
    if n_parse_ok == 0:
        raise SystemExit("无可用域，退出")

    # 校准（可选，force global）
    cal_slope = cal_off = None
    if ARGS.calibration_file:
        cal_slope, cal_off, mode, _ = load_calibration(ARGS.calibration_file, "GLOBAL_FORCE",
                                                       force_global=True)
        print(f"校准表: {mode} slope={cal_slope:.3f} off={cal_off:.3f}", flush=True)

    # backbone 只加载一次（所有 epoch 共用）
    model = load_model(ARGS.weights, device, model_type="auto")
    print(f"backbone 已加载: {Path(ARGS.weights).name}", flush=True)

    epochs = list(range(ARGS.start_epoch, ARGS.end_epoch + 1, ARGS.epoch_step))
    # 保证终态 epoch 一定在曲线内（v12.2=30 / v12.3=40 / v14=50）
    if epochs[-1] != ARGS.end_epoch:
        epochs.append(ARGS.end_epoch)
    results = {"meta": {"tag": ARGS.tag, "n_base": len(pick_base),
                        "n_supp": n_supp_actual, "n_per": ARGS.n_per,
                        "seed": ARGS.seed, "calibration_file": ARGS.calibration_file,
                        "labels": ARGS.labels, "supp_labels": ARGS.supp_labels,
                        "epochs": epochs}, "epochs": {}}

    for ep in epochs:
        ckpt = Path(ARGS.ckpt_dir) / f"finetune_epoch{ep:03d}.pt"
        if not ckpt.is_file():
            print(f"  !! 缺 {ckpt}，跳过", flush=True)
            continue
        enc = load_condition_encoder(str(ckpt), device)
        all_dev, all_hit, all_rec = [], [], []
        per_dom = {}
        for item_idx, item in enumerate(feats_list):
            did, L = item["did"], item["L"]
            fd = item["fd"]
            dom_res = {"L": L, "native_charge": round(float(net_charge(item["native"], 7.4)), 2), "arms": {}}
            for (j, (pH, tgt)) in enumerate(item["arms"]):
                tgt_eff = tgt
                if cal_slope is not None:
                    tgt_eff = (tgt - cal_off) / cal_slope
                charges, recs = [], []
                for k in range(ARGS.n_per):
                    torch.manual_seed(ARGS.seed * 1000 + item_idx * 100 + j * ARGS.n_per + k)
                    fd["randn"] = torch.randn(1, L)
                    cond_vec = make_condition_vector(pH, net_charge=tgt_eff)
                    out = conditioned_sample(model, enc, fd, cond_vec, device)
                    seq = seq_to_string(out["S"][0].cpu().numpy())
                    charges.append(float(net_charge(seq, pH)))
                    recs.append(sum(a == b for a, b in zip(seq, item["native"])) / L)
                mean_q = float(np.mean(charges))
                dev = abs(mean_q - tgt)
                hit = dev <= 2.0
                all_dev.append(dev); all_hit.append(hit); all_rec.append(float(np.mean(recs)))
                dom_res["arms"][f"ph_{pH:.1f}_t{tgt}"] = {"target": tgt, "mean_charge": round(mean_q, 2),
                                                          "dev": round(dev, 2), "hit": hit,
                                                          "recovery": round(float(np.mean(recs)), 3)}
            per_dom[did] = dom_res
        stats = {
            "n_domains_ok": len(feats_list),
            "n_arms": len(all_dev),
            "hit_rate": round(sum(all_hit) / len(all_hit), 3) if all_hit else None,
            "mean_dev": round(float(np.mean(all_dev)), 3) if all_dev else None,
            "mean_recovery": round(float(np.mean(all_rec)), 3) if all_rec else None,
        }
        results["epochs"][str(ep)] = stats
        print(f"epoch {ep:3d}: H2={stats['hit_rate']} dev={stats['mean_dev']} "
              f"rec={stats['mean_recovery']} (arms={stats['n_arms']})", flush=True)

    with open(ARGS.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"已写 {ARGS.out}", flush=True)


if __name__ == "__main__":
    main()

"""v11 修复诊断：target → 生成电荷 响应曲线（证实/证伪"目标域外推"假说）。

背景（v10 泛化退化，2026-08-28 分析结论）：
  v10-MoMPNN 在负目标域生成电荷 ≈ 2×target（斜率 1.2~2.4）且全域负偏置（截距 −2.4~−5.7）；
  v10-Ligand 未崩。主因候选 = 训练 target 值域覆盖不足 → 编码器在验证靶区（−19~−35）外推。
  本脚本用**现有 checkpoint**（不必重训）扫描 target 网格，画每条蛋白的
  "target → 生成电荷均值"响应曲线并拟合斜率/截距：

    · 若"训练域内蛋白"斜率≈1、而"验证域深负蛋白"斜率≈2 → 外推假说坐实（根因闭环）；
    · 若训练域内也斜率≈2 → 问题在模型响应本身（B/C 叠加），需换修法；
    · 若训练域/验证域都≈1（很不可能）→ 需要复查数据/配方。

用法（M oMPNN 侧首选；confumpnn 环境，在 code/ 平级目录下运行）：
  PYTHONPATH=code python v10_repair/v10_diag_response_curve.py \
      --cond_encoder output/finetune_v10_mompnn/finetune_epoch030.pt \
      --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt \
      --backbone auto \
      --manifest data/validation_pdbs/validation_manifest.json \
      --pdb-list /tmp/diag_training_domains.txt \
      --targets "-30,-25,-20,-15,-10,-5,0,5,10,15" \
      --n 20 --seed 3000 --pH 7.4 \
      --out output/v10_diag_response.json

  说明：
  - --manifest = 验证集 10 蛋白（OOD 侧）；--pdb-list = 一行一个 .pdb 路径文本文件
    （训练域侧；建议取 data/cath/S40/dompdb_pdb 里 5~8 个小/中域 + 1 个 native<-15 的深负域）。
  - --include_native 时会在每个蛋白的网格里追加它自己的 native 电荷点。
  - 输出 JSON：每蛋白 {slope, intercept, r2, targets, mean_charge, dev, ...}；
    终端打印汇总表 + "训练域 vs 验证域 斜率均值±std"。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# 从脚本位置向上搜索项目根（含 code/ 与 LigandMPNN/ 的目录），
# 使其无论放在 index/v10_repair/ 还是 code/tests/ 都能正确定位。
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


def linfit(xs, ys):
    """最小二乘 y = a*x + b；返回 (a, b, r2)。"""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    a = sxy / sxx if sxx > 1e-9 else float("nan")
    b = my - a * mx
    r2 = (sxy ** 2 / (sxx * syy)) if sxx > 1e-9 and syy > 1e-9 else float("nan")
    return a, b, r2


def main():
    ap = argparse.ArgumentParser(description="v11 目标→生成电荷 响应曲线诊断")
    ap.add_argument("--cond_encoder", required=True, help="条件编码器 checkpoint")
    ap.add_argument("--weights", required=True, help="backbone 权重")
    ap.add_argument("--backbone", default="auto",
                    choices=["auto", "protein_mpnn", "ligand_mpnn"])
    ap.add_argument("--manifest", default=None,
                    help="验证集 manifest JSON（OOD 蛋白，可选）")
    ap.add_argument("--pdb", action="append", default=[],
                    help="单个 .pdb 路径（可重复）")
    ap.add_argument("--pdb-list", default=None,
                    help="一行一个 .pdb 路径的文本文件（训练域侧/任意侧）")
    ap.add_argument("--targets", required=True,
                    help="target 网格，逗号分隔：'-30,-25,...,15'")
    ap.add_argument("--include_native", action="store_true",
                    help="每个蛋白追加其 native 电荷点")
    ap.add_argument("--pH", type=float, default=7.4)
    ap.add_argument("--n", type=int, default=20, help="每 target 采样条数")
    ap.add_argument("--seed_base", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_ligand_atoms", type=int, default=16)
    ap.add_argument("--out", required=True, help="结果 JSON 路径")
    args = ap.parse_args()

    device = torch.device(args.device)
    targets = [float(x) for x in args.targets.split(",") if x.strip() != ""]
    print(f"target 网格: {targets}  n={args.n}  pH={args.pH}  "
          f"backbone={args.backbone}", flush=True)

    # ---- 收集蛋白列表 ----
    pdbs = []  # [(name, path, group)]
    for p in args.pdb:
        pdbs.append((Path(p).stem, Path(p), "pdb"))
    if args.pdb_list:
        with open(args.pdb_list) as f:
            for line in f:
                line = line.strip()
                if line:
                    pdbs.append((Path(line).stem, Path(line), "trainish"))
    if args.manifest:
        man = json.load(open(args.manifest))
        for it in man["items"]:
            pdbs.append((it["pdb"], Path(it["path"]), "valid"))
    if not pdbs:
        raise SystemExit("没有蛋白输入：--pdb / --pdb-list / --manifest 至少给一个")

    # ---- 加载模型 ----
    enc = load_condition_encoder(args.cond_encoder, device)
    model = load_model(args.weights, device, model_type=args.backbone)
    backbone_type = model.model_type
    print(f"backbone 实际类型: {backbone_type}", flush=True)
    if backbone_type == "protein_mpnn":
        feats = dict(model_type="protein_mpnn", use_atom_context=False,
                     number_of_ligand_atoms=0)
    else:
        feats = dict(model_type="ligand_mpnn", use_atom_context=True,
                     number_of_ligand_atoms=args.num_ligand_atoms)

    results = {"meta": {"cond_encoder": args.cond_encoder, "weights": args.weights,
                        "pH": args.pH, "n": args.n, "targets": targets},
               "proteins": {}}
    print(f"\n{'name':8s} {'L':>4s} {'nativeQ':>7s} {'slope':>6s} {'int':>6s} {'r2':>5s} "
          f"{'group':8s} |  target→均值 示例", flush=True)

    for name, pdb_path, group in pdbs:
        try:
            protein_dict, *_ = parse_PDB(str(pdb_path))
        except Exception as e:
            print(f"  !! {name} parse 失败: {e}", flush=True)
            continue
        L = protein_dict["X"].shape[0]
        native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
        q_nat = float(net_charge(native, args.pH))
        protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
        fd = featurize(protein_dict, cutoff_for_score=8.0, **feats)
        fd["batch_size"] = 1
        fd["temperature"] = args.temperature
        fd["bias"] = torch.zeros(1, L, 21)
        fd["randn"] = torch.randn(1, L)

        tgt_list = list(targets)
        if args.include_native:
            tgt_list.append(round(q_nat))
        means, devs, recs = [], [], []
        for tgt in tgt_list:
            charges, rec_tmp = [], []
            for k in range(args.n):
                torch.manual_seed(args.seed_base + k)
                fd["randn"] = torch.randn(1, L)
                cond_vec = make_condition_vector(args.pH, net_charge=tgt)
                out = conditioned_sample(model, enc, fd, cond_vec, device)
                seq = seq_to_string(out["S"][0].cpu().numpy())
                charges.append(float(net_charge(seq, args.pH)))
                rec_tmp.append(sum(a == b for a, b in zip(seq, native)) / L)
            means.append(float(np.mean(charges)))
            devs.append(abs(float(np.mean(charges)) - tgt))
            recs.append(float(np.mean(rec_tmp)))
        slope, inter, r2 = linfit(tgt_list, means)
        results["proteins"][name] = {
            "group": group, "L": L, "native_charge": round(q_nat, 2),
            "targets": [round(t, 1) for t in tgt_list],
            "mean_charge": [round(m, 2) for m in means],
            "dev": [round(d, 2) for d in devs],
            "recovery": [round(r, 3) for r in recs],
            "slope": round(slope, 3), "intercept": round(inter, 3), "r2": round(r2, 3),
        }
        demo = f"{tgt_list[0]:+.0f}→{means[0]:+.1f} ... {tgt_list[-1]:+.0f}→{means[-1]:+.1f}"
        print(f"{name:8s} {L:4d} {q_nat:7.1f} {slope:6.2f} {inter:6.1f} {r2:5.2f} "
              f"{group:8s} | {demo}", flush=True)

    # ---- 汇总：训练域 vs 验证域斜率 ----
    print("\n=== 斜率汇总（按 group）===")
    from collections import defaultdict
    by_g = defaultdict(list)
    for v in results["proteins"].values():
        by_g[v["group"]].append(v["slope"])
    for g, vals in by_g.items():
        m = float(np.mean(vals)); s = float(np.std(vals)) if len(vals) > 1 else float("nan")
        print(f"  {g:8s}: n={len(vals):2d}  slope 均值={m:.2f} ± {s:.2f}"
              + ("   ← 若训练域≈1 而验证域≈2，外推假说坐实" if g in ("trainish", "valid") else ""))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已写: {args.out}")


if __name__ == "__main__":
    main()

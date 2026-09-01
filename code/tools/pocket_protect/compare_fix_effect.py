"""口袋 fix 效果对比：加口袋 fix（--fixed_residues）前后，配体口袋删减是否缓解。

背景（2026-09-01 配体模式删减捷径）：
  v12 微调教会配体模式"删带电残基调电荷"，深部口袋（frac_sasa<0.25）带电残基
  不在任何损失监督内 → 成对删逃逸（10 蛋白 34 建议 fix 位点 27 实际被删=79%）。
  本脚本对比同一蛋白 native 臂：无 fix（泛化验证已生成）vs 加口袋 fix
  （define_pocket.py 输出的 pocket_fix.txt 喂 --fixed_residues 重新生成）。

口径：
  - 带电残基 = D/E + K/R（与 compare_comp_ligand 一致）
  - 口袋 = 配体 8Å 内（pocket_table.json 的 pocket rows）
  - fix 位点 = pocket_table.json level=建议fix(深部带电) 的 idx
  - 倍率 = 生成均值 / native 计数；删除 = 1 − 倍率（<0 为删减）
  - charge dev = |生成平均电荷 − target|（H2 判据 ≤2.0）

用法（项目根）：
  /home/baokun_yu/miniconda3/envs/confumpnn/bin/python \
      code/tools/pocket_protect/compare_fix_effect.py \
      --name 2FEO --pdb data/validation_pdbs/2FEO.pdb \
      --no-fix-root output/generalization_ligand_v12_2/ligand \
      --fix-root output/pocket_fix_test/v12_2/ligand
输出：终端表 + output/pocket_fix_test/compare_fix_effect.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_DIR = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
sys.path.insert(0, str(_PROJECT_DIR / "LigandMPNN"))
sys.path.insert(0, str(_PROJECT_DIR / "code"))
from run_guided import net_charge  # noqa: E402

CHARGED = "DEKR"


def read_seqfa(fa):
    """读 seqs.fa：返回 (生成序列列表, native 序列)。"""
    lines = open(fa).read().splitlines()
    seqs, cur = [], None
    for line in lines:
        if line.startswith(">"):
            if cur is not None:
                seqs.append(cur)
            cur = (line[1:], "")
        elif cur is not None:
            cur = (cur[0], cur[1] + line)
    if cur is not None:
        seqs.append(cur)
    gen = [s for n, s in seqs if not n.startswith("native")]
    nat = [s for n, s in seqs if n.startswith("native")]
    return gen, (nat[0] if nat else None)


def charged_in(seq, idx):
    return sum(1 for i in idx if seq[i] in CHARGED)


def mean(arr):
    return sum(arr) / len(arr) if arr else float("nan")


def load_pocket(name):
    tab = json.load(open(_PROJECT_DIR / "output/pocket_protect" / name / "pocket_table.json"))
    pocket_idx = np.array([r["idx"] for r in tab["pocket"]], dtype=int)
    fix_idx = np.array([r["idx"] for r in tab["pocket"]
                        if r["level"] == "建议fix(深部带电)"], dtype=int)
    return pocket_idx, fix_idx


def analyze(fa, pocket_idx, fix_idx, pH):
    gen, native = read_seqfa(fa)
    if not gen or not native:
        return None
    L = len(native)
    nat_dk = charged_in(native, range(L))
    nat_pkt = charged_in(native, pocket_idx)
    nat_fix = charged_in(native, fix_idx)
    g_dk = [charged_in(s, range(L)) for s in gen]
    g_pkt = [charged_in(s, pocket_idx) for s in gen]
    g_fix = [charged_in(s, fix_idx) for s in gen]
    charges = [float(net_charge(s, pH)) for s in gen]
    rec = [sum(a == b for a, b in zip(s, native)) / L for s in gen]
    rec_pkt = [sum(s[i] == native[i] for i in pocket_idx) / len(pocket_idx) for s in gen]
    # target：从 fasta header 的 target= 读（native 臂 = native 电荷）
    target = None
    for line in open(fa):
        if line.startswith(">seed") and "target=" in line:
            target = float(line.split("target=")[1].split()[0])
            break
    return {
        "native_dk": nat_dk, "gen_dk": round(mean(g_dk), 1),
        "dk_ratio": round(mean(g_dk) / nat_dk, 2) if nat_dk else None,
        "native_pkt": nat_pkt, "gen_pkt": round(mean(g_pkt), 1),
        "pkt_ratio": round(mean(g_pkt) / nat_pkt, 2) if nat_pkt else None,
        "native_fix": nat_fix, "gen_fix": round(mean(g_fix), 1),
        "fix_ratio": round(mean(g_fix) / nat_fix, 2) if nat_fix else None,
        "charge": round(mean(charges), 2), "target": round(target, 2) if target else None,
        "dev": round(abs(mean(charges) - target), 2) if target else None,
        "recovery": round(mean(rec), 3), "recovery_pkt": round(mean(rec_pkt), 3),
        "n": len(gen),
    }


def main():
    ap = argparse.ArgumentParser(description="口袋 fix 效果对比（删减是否缓解）")
    ap.add_argument("--names", default="1C6O,1AXW,2FEO",
                    help="逗号分隔蛋白名（manifest 顺序，或用 --pdb）")
    ap.add_argument("--no-fix-root", default=str(_PROJECT_DIR / "output/generalization_ligand_v12_2/ligand"),
                    help="无 fix 生成根目录（含 <name>/pH7.4/arm_native/seqs.fa）")
    ap.add_argument("--fix-root", default=str(_PROJECT_DIR / "output/pocket_fix_test/v12_2/ligand"),
                    help="加 fix 生成根目录")
    ap.add_argument("--pH", type=float, default=7.4)
    args = ap.parse_args()

    names = [n for n in args.names.split(",") if n.strip()]
    print(f"{'蛋白':6s} {'口径':>4s} {'native':>6s} {'gen':>6s} {'倍率':>5s} "
          f"{'删减':>5s} {'charge':>7s} {'dev':>5s} {'rec':>5s} {'rec_pkt':>6s}", flush=True)
    results = {}
    for name in names:
        pocket_idx, fix_idx = load_pocket(name)
        rows = []
        for tag, root in [("无fix", args.no_fix_root), ("有fix", args.fix_root)]:
            fa = Path(root) / name / f"pH{args.pH}" / "arm_native" / "seqs.fa"
            if not fa.exists():
                print(f"  !! {name} 缺 {fa}")
                continue
            a = analyze(str(fa), pocket_idx, fix_idx, args.pH)
            if a is None:
                print(f"  !! {name} 序列读取失败")
                continue
            a["tag"] = tag
            rows.append(a)
        if len(rows) < 2:
            continue
        base, fixed = rows[0], rows[1]
        print(f"{name:6s}", flush=True)
        for a in rows:
            r = a["pkt_ratio"]
            if r is None:
                del_pct = "  -"
            elif r >= 1:
                del_pct = f"+{(r-1)*100:.0f}%增"
            else:
                del_pct = f"-{(1-r)*100:.0f}%删"
            print(f"     {a['tag']:>4s} 全 {a['native_dk']:3d}/{a['gen_dk']:5.1f} "
                  f"{a['dk_ratio']:5.2f} {del_pct:>5s} "
                  f"charge {a['charge']:+6.1f}(t{a['target']:+.0f}) dev {a['dev']:4.1f} "
                  f"rec {a['recovery']:.3f} {a['recovery_pkt']:.3f}", flush=True)
        # 深部 fix 位点（专门看）
        print(f"     [深部fix位点] 无fix {base['native_fix']}->{base['gen_fix']} "
              f"({base['fix_ratio']}×) | 有fix {fixed['native_fix']}->{fixed['gen_fix']} "
              f"({fixed['fix_ratio']}×)", flush=True)
        results[name] = {"no_fix": base, "with_fix": fixed}

    out = _PROJECT_DIR / "output/pocket_fix_test"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "compare_fix_effect.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n已写 {out}/compare_fix_effect.json", flush=True)


if __name__ == "__main__":
    main()

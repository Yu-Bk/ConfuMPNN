"""配体模式组成分析：泛化验证 native 臂生成序列 vs native 的 D/E/K/R 计数。

治删减捷径（模型无差别删带电残基）与过度添加。与 mompnn compare_comp_v12_2.py
口径一致（带电残基总数 = D/E + K/R），但直接读泛化验证已生成的 native 臂序列
（<gen-root>/<pdb>/pH7.4/arm_native/seqs.fa），不重新采样。target=native 臂即
"target 条件=自身 native 电荷"的生成。

v12.2 与 v13 共用（路径参数化）：--gen-root 指定泛化验证根（ligand/ 层），
--out 指定输出 JSON。

用法（项目根）：
  PYTHONPATH=code python code/tests/ligand_v9/compare_comp_ligand.py \
      --gen-root output/generalization_ligand_v12_2/ligand --out output/v12_2_ligand_comp.json
  # v13
  PYTHONPATH=code python code/tests/ligand_v9/compare_comp_ligand.py \
      --gen-root output/generalization_ligand_v13/ligand --out output/v13_ligand_comp.json
"""
import argparse
import json
import sys
from pathlib import Path

_PROJECT_DIR = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
MANIFEST = _PROJECT_DIR / "data/validation_pdbs/validation_manifest.json"


def count_dk(seq):
    """带电残基总数 = D/E + K/R（与 v12.1/v12.2 报告口径一致）。"""
    de = sum(1 for a in seq if a in "DE")
    kr = sum(1 for a in seq if a in "KR")
    return de + kr, de, kr


def read_seqfa(fa):
    """读 seqs.fa：返回 [(name, seq), ...]（跳过 native 行，只统计生成序列）。"""
    seqs, cur = [], None
    for line in open(fa):
        line = line.strip()
        if line.startswith(">"):
            if cur is not None:
                seqs.append(cur)
            cur = (line[1:], "")
        elif cur is not None:
            cur = (cur[0], cur[1] + line)
    if cur is not None:
        seqs.append(cur)
    # 去掉 native 行（header 含 'charge=' 的是生成；native 行是 '>native charge=...'）
    return [(n, s) for n, s in seqs if not n.startswith("native")]


def main():
    ap = argparse.ArgumentParser(description="配体模式组成分析（v12.2/v13 共用）")
    ap.add_argument("--gen-root", default=str(_PROJECT_DIR / "output/generalization_ligand_v12_2" / "ligand"),
                    help="泛化验证根（ligand/ 层，含 <pdb>/pH7.4/arm_native/seqs.fa）")
    ap.add_argument("--out", default=str(_PROJECT_DIR / "output" / "v12_2_ligand_comp.json"),
                    help="输出 JSON 路径")
    args = ap.parse_args()
    gen_root = Path(args.gen_root)
    man = json.load(open(MANIFEST))
    items = man["items"]
    print(f"{'name':7s} {'L':>4s} {'native_DK':>9s} {'gen_DK':>8s} {'倍率':>5s} "
          f"{'native_DE/KR':>13s} {'gen_DE/KR':>12s}", flush=True)
    results = {}
    for it in items:
        name = it["pdb"]
        fa = gen_root / name / "pH7.4" / "arm_native" / "seqs.fa"
        if not fa.exists():
            print(f"  !! {name} 无 native 臂 seqs.fa", flush=True)
            continue
        # native 序列：从生成 fasta 的 native 行后读
        native = None
        lines = open(fa).read().splitlines()
        for i, line in enumerate(lines):
            if line.startswith(">native") and i + 1 < len(lines):
                native = lines[i + 1].strip()
                break
        if native is None:
            print(f"  !! {name} 无 native 序列行", flush=True)
            continue
        gen = read_seqfa(fa)
        if not gen:
            print(f"  !! {name} 无生成序列", flush=True)
            continue

        L = len(native)
        nat_dk, nat_de, nat_kr = count_dk(native)
        gdk, gde, gkr = [], [], []
        for _, s in gen:
            dk, de, kr = count_dk(s)
            gdk.append(dk); gde.append(de); gkr.append(kr)
        m_dk = sum(gdk) / len(gdk)
        m_de = sum(gde) / len(gde)
        m_kr = sum(gkr) / len(gkr)
        ratio = m_dk / nat_dk if nat_dk else float("nan")
        print(f"{name:7s} {L:4d} {nat_dk:9d} {m_dk:8.1f} {ratio:5.2f} "
              f"{nat_de}/{nat_kr:>6d} {m_de:7.1f}/{m_kr:6.1f}", flush=True)
        results[name] = {"L": L, "native_dk": nat_dk, "native_de": nat_de,
                         "native_kr": nat_kr, "gen_dk": round(m_dk, 1),
                         "gen_de": round(m_de, 1), "gen_kr": round(m_kr, 1),
                         "ratio": round(ratio, 2), "n_gen": len(gen)}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n已写 {args.out}", flush=True)


if __name__ == "__main__":
    main()

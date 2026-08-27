"""PROPKA 微环境电荷复核（v3 方案 P5 / 判据 H4）。

目的：把"设计电荷 Q_design（游离 pKa + Henderson-Hasselbalch）"与
"物理修正电荷 Q_phys（PROPKA3 微环境修正 pKa 后重算）"对照，检验设计电荷
的**物理真实性**。

原理（v3 §3.3）：
  1. 生成序列 → ESMFold/AF2 回折结构（PDB）
  2. PROPKA3 对每条结构计算微环境修正 pKa（侧链 + N/C 端）
  3. 用修正 pKa 重算净电荷 = 物理修正电荷 Q_phys
  4. 对照设计电荷 Q_design（游离 pKa，来自 differentiable_charge.net_charge）
  5. 新判据 H4：|Q_phys 均值 − target| ≤ 2.0 的臂达标率

用法（confumpnn 环境，PROPKA 3.5.1 已装）：
  # 单条回折结构
  python propka_charge_check.py --pdb output/fold/sample_1.pdb --pH 7.4 --target -2.0

  # 一个臂的目录（含 n 条回折 PDB，逐个算并汇总均值 ± std）
  python propka_charge_check.py --pdb output/generalization_v9/protein/1MBN/pH7.4/arm_n2 \
      --pH 7.4 --target -2.0

输出：
  - 打印每条结构的 Q_design / Q_phys
  - 打印均值 ± std、H4 判定（|Q_phys 均值 − target| ≤ 2.0）
  - --out 时写 JSON

说明（诚实边界，写论文时引用）：
  - PROPKA 只报告可滴定残基（D/E/H/C/Y/K/R + N/C 端）；其余残基电荷计 0。
  - 设计是**组成层面**电荷（游离 pKa），非微环境精确 pKa 设计；H4 检验的是
    二者差异是否在阈值内。部分臂在物理口径下"失败"正是物理边界证据。
"""

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# 让本脚本能 import src.*（配合 PYTHONPATH=. 或自动添加）
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from src.differentiable_charge import net_charge  # noqa: E402
from src.pka import AA_TO_IDX  # noqa: E402

LN10 = math.log(10.0)

# PROPKA Group 名 → 单字母氨基酸
GROUP_TO_AA = {
    "ASP": "D", "GLU": "E", "HIS": "H", "CYS": "C",
    "TYR": "Y", "LYS": "K", "ARG": "R",
}
ACIDIC_AA = ("D", "E", "C", "Y")   # 去质子化带 -1
BASIC_AA = ("K", "R", "H")         # 质子化带 +1
TERM_GROUPS = {"N+": "N", "C-": "C"}  # PROPKA 的 N 端/C 端组名

# 从 PDB 文件提取序列（只取第一个链；PROPKA 会处理整个文件的所有链，
# 但回折结构通常单链，取主链做序列对照）
_RESNAME_TO_AA = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def extract_seq_from_pdb(pdb_path):
    """从 PDB 提取第一条**蛋白链**的氨基酸序列（单字母字符串）。

    只取标准 20 氨基酸的 ATOM 记录（跳过核酸/配体/非标准残基）；
    同链内残基号变化才 append 一个字母（避免把同残基的多个原子算多次）；
    收集完第一条蛋白链后遇到新链即停止。适合 1BC8 这类 DNA-蛋白复合物。
    """
    seq = []
    cur_chain = None
    prev_resid = None
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            resname = line[17:20].strip()
            aa = _RESNAME_TO_AA.get(resname)
            if aa is None:
                continue  # 跳过核酸/非标准残基
            if cur_chain is None:
                cur_chain = chain  # 从第一条蛋白链开始
            if chain != cur_chain:
                if seq:
                    break  # 第一条蛋白链已收集完
                cur_chain = chain
            resid = (int(line[22:26]), line[26])  # (resseq, icode)
            if resid != prev_resid:
                seq.append(aa)
                prev_resid = resid
    return "".join(seq)


def run_propka(pdb_path, workdir):
    """调用 propka3 CLI 生成 .pka 文件，返回解析出的残基 pKa 表。

    propka3 把输出写到**当前工作目录**（本版本不支持 --output 参数），
    因此把输入 pdb 复制到临时工作目录并 cd 进去运行，生成的 .pka 就在原地。

    返回: list of dict {aa, resnum, chain, pka}（只含可滴定残基 + 端基）。
    """
    stem = Path(pdb_path).stem
    in_tmp = Path(workdir) / Path(pdb_path).name
    shutil.copy2(pdb_path, in_tmp)
    p = subprocess.run(
        ["propka3", in_tmp.name],
        cwd=workdir, capture_output=True, text=True, timeout=300,
    )
    pka_file = Path(workdir) / f"{stem}.pka"
    if not pka_file.exists():
        raise RuntimeError(
            f"PROPKA 未生成 .pka（{pka_file}）\nstdout: {p.stdout[-500:]}"
        )

    groups = []
    in_summary = False
    with open(pka_file) as f:
        for line in f:
            if "SUMMARY OF THIS PREDICTION" in line:
                in_summary = True
                continue
            if not in_summary:
                continue
            toks = line.split()
            if len(toks) < 5:
                continue
            gname = toks[0]
            if gname in GROUP_TO_AA or gname in TERM_GROUPS:
                try:
                    pka_val = float(toks[3])
                except ValueError:
                    continue
                resnum = int(toks[1])
                chain = toks[2]
                groups.append({
                    "aa": GROUP_TO_AA.get(gname, gname),
                    "gname": gname,
                    "resnum": resnum,
                    "chain": chain,
                    "pka": pka_val,
                })
    return groups


def phys_charge_from_pka(groups, pH, seq):
    """用 PROPKA 修正 pKa 重算净电荷 Q_phys。

    规则与 differentiable_charge.py 一致（HH 方程），只是 pKa 换成微环境修正值：
      酸性 D/E/C/Y：去质子化分数 = 1/(1+10^(pKa−pH))，Q = -1·分数
      碱性 K/R/H  ：质子化分数   = 1/(1+10^(pH−pKa))，Q = +1·分数
      N 端（+1）  ：质子化分数   = 1/(1+10^(pH−pKa))
      C 端（-1）  ：去质子化分数 = 1/(1+10^(pKa−pH))
    未列出的残基贡献 0。

    参数:
        groups: run_propka 返回的残基表
        pH: 工作 pH
        seq: 序列字符串（占位，保留签名，未使用）
    返回: float
    """
    total = 0.0
    for g in groups:
        aa = g["aa"]
        pka = g["pka"]
        if aa == "N":  # N 端（+1），质子化分数 = 1/(1+10^(pH-pKa))
            total += 1.0 / (1.0 + 10 ** (pH - pka))
        elif aa in ACIDIC_AA:
            # 酸性 D/E/C/Y + C 端（均 -1），去质子化分数 = 1/(1+10^(pKa-pH))
            total += -1.0 / (1.0 + 10 ** (pka - pH))
        elif aa in BASIC_AA:
            total += 1.0 / (1.0 + 10 ** (pH - pka))
    return total


def check_single(pdb_path, pH, target, workdir):
    """对单条回折结构计算 Q_design / Q_phys，返回 dict。"""
    seq = extract_seq_from_pdb(pdb_path)
    q_design = net_charge(seq, pH) if seq else None
    groups = run_propka(pdb_path, workdir)
    q_phys = phys_charge_from_pka(groups, pH, seq)
    return {
        "pdb": str(pdb_path),
        "seq": seq,
        "L": len(seq) if seq else 0,
        "q_design": round(q_design, 3) if q_design is not None else None,
        "q_phys": round(q_phys, 3),
        "n_titratable_reported": len(groups),
    }


def main():
    ap = argparse.ArgumentParser(description="PROPKA 微环境电荷复核（H4）")
    ap.add_argument("--pdb", required=True,
                    help="单条回折 PDB 或含多条 PDB 的目录（臂目录）")
    ap.add_argument("--pH", type=float, required=True, help="工作 pH")
    ap.add_argument("--target", type=float, default=None,
                    help="该臂的目标净电荷（用于 H4 判定）")
    ap.add_argument("--out", default=None, help="输出 JSON 路径（可选）")
    args = ap.parse_args()

    p = Path(args.pdb)
    if p.is_dir():
        pdbs = sorted(glob.glob(str(p / "*.pdb")))
        label = f"目录 {p.name}（{len(pdbs)} 条结构）"
    else:
        pdbs = [str(p)]
        label = f"单条 {p.name}"

    print(f"PROPKA 复核: {label}  pH={args.pH}", flush=True)
    with tempfile.TemporaryDirectory() as workdir:
        results = [check_single(pdb, args.pH, args.target, workdir)
                   for pdb in pdbs]

    print(f"{'pdb':<40}{'L':>4}{'Q_design':>9}{'Q_phys':>9}", flush=True)
    for r in results:
        print(f"{Path(r['pdb']).name:<40}{r['L']:>4}"
              f"{str(r['q_design']):>9}{r['q_phys']:>9}", flush=True)

    qp = [r["q_phys"] for r in results if r["q_phys"] is not None]
    qd = [r["q_design"] for r in results if r["q_design"] is not None]
    out = {
        "pdb": args.pdb, "pH": args.pH, "target": args.target,
        "n": len(results),
        "results": results,
        "q_phys_mean": round(float(np.mean(qp)), 3) if qp else None,
        "q_phys_std": round(float(np.std(qp)), 3) if qp else None,
        "q_design_mean": round(float(np.mean(qd)), 3) if qd else None,
    }
    if args.target is not None and qp:
        dev = abs(out["q_phys_mean"] - args.target)
        out["h4_dev"] = round(dev, 3)
        out["h4_pass"] = dev <= 2.0
        print(f"\nQ_phys 均值 = {out['q_phys_mean']} ± {out['q_phys_std']}  "
              f"(Q_design 均值 = {out['q_design_mean']})", flush=True)
        print(f"H4 判定: |Q_phys − target| = {dev:.3f} → "
              f"{'PASS' if dev <= 2.0 else 'FAIL'}"
              f"（阈值 2.0，与 H2 一致）", flush=True)
    else:
        print(f"\nQ_phys 均值 = {out['q_phys_mean']} ± {out['q_phys_std']}  "
              f"（未给 --target，不判定 H4）", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"已写 {args.out}", flush=True)


if __name__ == "__main__":
    main()

"""v9 配体训练数据获取：RCSB 候选池 → 并行下载 → 本地五类配体分类 + 长度过滤。

背景（index/PROJECT_V9_LIGAND_PLAN.md）：在 LigandMPNN backbone 上重训条件编码器，
需要五类配体复合物（RNA/DNA/小分子/金属/多结合水），总量 ~9000，比 MoMPNN(7886) 稍多。

三步：
  1. 采样（Python）：RCSB 搜索 API 分页拉候选池 → 随机采样 N 个
  2. 下载（xargs 并行 curl）：pdb 优先，404 降级 cif，保留真实后缀
  3. 分类（Python）：本地 HETATM 解析 → 过滤 L≤300 → 按残基名分五类 → 各取目标数

用法（base 环境，需 requests）：
  # 1) 采样 + 下载 + 分类（一步）
  python tests/fetch_ligand_pdbs.py --sampled 15000 --out data/ligand_train \
      --targets rna:1000,dna:1000,small_mol:3000,metal:2500,water:1500 --parallel 12
  # 2) 只采样（生成 candidates.json）
  python tests/fetch_ligand_pdbs.py --sampled 15000 --out data/ligand_train --stage sample
  # 3) 只下载（用已有 candidates.json）
  python tests/fetch_ligand_pdbs.py --out data/ligand_train --stage download --parallel 12
  # 4) 只分类
  python tests/fetch_ligand_pdbs.py --out data/ligand_train --stage classify --targets ...

输出：
  data/ligand_train/{rna,dna,small_mol,metal,water}/<id>.pdb|cif  五类 PDB
  data/ligand_train/candidates.json      候选 ID 清单
  data/ligand_train/classification.json  分类元数据（含被过滤原因）
  data/ligand_train/all_pdb/             五类合并（symlink，供训练 --dompdb）
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import requests

RCSB_QUERY = {
    'query': {'type': 'group', 'logical_operator': 'and', 'nodes': [
        {'type': 'terminal', 'service': 'text', 'parameters': {
            'attribute': 'rcsb_entry_info.nonpolymer_entity_count',
            'operator': 'greater_or_equal', 'value': 1}},
        {'type': 'terminal', 'service': 'text', 'parameters': {
            'attribute': 'rcsb_entry_info.polymer_entity_count_protein',
            'operator': 'equals', 'value': 1}},
        {'type': 'terminal', 'service': 'text', 'parameters': {
            'attribute': 'rcsb_entry_info.polymer_entity_count_nucleic_acid',
            'operator': 'equals', 'value': 0}},
        {'type': 'terminal', 'service': 'text', 'parameters': {
            'attribute': 'rcsb_entry_info.polymer_entity_count',
            'operator': 'equals', 'value': 1}},
        {'type': 'terminal', 'service': 'text', 'parameters': {
            'attribute': 'rcsb_entry_info.resolution_combined',
            'operator': 'less_or_equal', 'value': 2.5}},
    ]},
    'return_type': 'entry',
}

# ---- 配体分类规则（本地 HETATM 残基名判定）----
RNA_RESIDUES = {
    "AMP", "ADP", "ATP", "GMP", "GDP", "GTP", "CMP", "CDP", "CTP",
    "UMP", "UDP", "UTP", "URA", "ADE", "GUA", "CYT", "RGU", "RCY",
    "1MA", "1MG", "2MG", "5MC", "7MG", "M2G", "OMC", "OMG", "PSU",
    "2BU", "4OC", "5BU", "H2U", "I", "YG", "YYG", "OMU",
}
DNA_RESIDUES = {
    "DA", "DC", "DG", "DT", "DU", "DI", "DAN", "DCN", "DGN", "DTN",
    "A", "C", "G", "T", "U",
}
METAL_ELEMENTS = {
    "ZN", "MG", "CA", "FE", "CU", "MN", "CO", "NI", "NA", "K", "CD",
    "HG", "PB", "MO", "W", "LI", "SR", "BA", "AL", "AG", "AU", "PT",
    "V", "CR", "SE", "RB", "CS", "GA", "GE", "SB", "BI", "SN", "CE",
    "PR", "ND", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB",
    "LU", "RU", "RH", "PD", "OS", "IR",
}
HOH = {"HOH", "WAT", "DOD"}


def parse_ligand_info(path):
    """扫描 PDB/cif HETATM，返回 (ligand_resnames, n_residues)。
    n_residues 从 CA 原子数估计（pdb 行或 cif 行）。
    """
    counts = {}
    n_ca = 0
    pdb = str(path).endswith(".pdb")
    with open(path) as f:
        for line in f:
            if pdb:
                if line.startswith("HETATM"):
                    resn = line[17:20].strip()
                    counts[resn] = counts.get(resn, 0) + 1
                elif line.startswith("ATOM") and line[12:16].strip() == "CA":
                    n_ca += 1
            else:  # cif
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    if "_atom_site" in line:
                        continue
    if not pdb:
        # cif 用简单统计：统计 HETATM 行 + CA 行
        n_ca = 0
        for line in open(path):
            if line.startswith("ATOM"):
                cols = line.split()
                if len(cols) > 2 and cols[2] == "CA":
                    n_ca += 1
    ligand = {k: v for k, v in counts.items() if k not in HOH}
    return ligand, n_ca


def classify_ligand(ligand_counts):
    """按 HETATM 残基名判定主配体类型。优先级 RNA > DNA > 小分子 > 金属 > 水。"""
    if not ligand_counts:
        return "water", "仅水/无配体"
    rna = [r for r in ligand_counts if r in RNA_RESIDUES]
    dna = [r for r in ligand_counts if r in DNA_RESIDUES and r not in RNA_RESIDUES]
    if rna:
        return "rna", f"RNA核苷酸 {rna[:4]}"
    if dna:
        return "dna", f"DNA核苷酸 {dna[:4]}"
    metal_names = set(ligand_counts.keys()) & METAL_ELEMENTS
    organic = [r for r in ligand_counts if r not in METAL_ELEMENTS]
    if metal_names and not organic:
        return "metal", f"纯金属 {sorted(metal_names)[:4]}"
    if organic:
        return "small_mol", f"有机 {organic[:4]}"
    if metal_names:
        return "metal", f"金属+杂 {sorted(metal_names)[:4]}"
    return "water", "其他"


def get_candidates(sampled):
    """RCSB 搜索 API 拉候选池（单次 rows=10000，实测 0.8s）→ 随机采样。"""
    q = dict(RCSB_QUERY)
    q['request_options'] = {'paginate': {'start': 0, 'rows': 10000}}
    r = requests.post('https://search.rcsb.org/rcsbsearch/v2/query', json=q, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"RCSB API 错误: {r.status_code} {r.text[:200]}")
    ids = [h['identifier'] for h in r.json().get('result_set', [])]
    print(f"候选池 {len(ids)} 个，随机采样 {sampled}", flush=True)
    random.seed(42)
    return random.sample(ids, min(sampled, len(ids)))


def download_pdbs(cands, out_dir, parallel=12):
    """xargs 并行 curl 下载。pdb 优先，404 降级 cif，保留真实后缀。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    id_file = out_dir / "_ids.txt"
    id_file.write_text("\n".join(cands))
    script = f'''#!/bin/bash
cat {id_file} | xargs -P {parallel} -I{{}} bash -c '
  id="{{}}"; d={out_dir}
  if [ -f "$d/$id.pdb" ] || [ -f "$d/$id.cif" ]; then exit 0; fi
  if curl -sfL --max-time 60 -o "$d/$id.pdb" "https://files.rcsb.org/download/$id.pdb" 2>/dev/null && [ -s "$d/$id.pdb" ]; then exit 0; fi
  rm -f "$d/$id.pdb"
  curl -sfL --max-time 60 -o "$d/$id.cif" "https://files.rcsb.org/download/$id.cif" 2>/dev/null && [ -s "$d/$id.cif" ] || rm -f "$d/$id.cif"
'
'''
    tmp = out_dir / "_download.sh"
    tmp.write_text(script)
    r = subprocess.run(["bash", str(tmp)], capture_output=True, text=True)
    n = sum(1 for p in out_dir.glob("*.pdb")) + sum(1 for p in out_dir.glob("*.cif"))
    print(f"下载完成 {n}/{len(cands)}（含 cif 降级）", flush=True)
    return n


def classify_and_pick(cands, out_dir, targets, max_len=300):
    """本地解析分类 + 过滤 L≤300 + 各取目标数。"""
    cat_dirs = {}
    for tok in targets.split(","):
        name, n = tok.split(":")
        cat_dirs[name] = int(n)
    for name in cat_dirs:
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    meta = {c: [] for c in cat_dirs}
    filtered = {"len": 0, "parse_fail": 0}
    for pid in cands:
        src = None
        for suf in (".pdb", ".cif"):
            p = out_dir / f"{pid}{suf}"
            if p.exists():
                src = p
                break
        if src is None:
            continue
        try:
            ligand, n_ca = parse_ligand_info(src)
        except Exception:
            filtered["parse_fail"] += 1
            continue
        if n_ca <= 0 or n_ca > max_len:
            filtered["len"] += 1
            continue
        cat, reason = classify_ligand(ligand)
        if cat not in cat_dirs:
            continue
        # 移入分类目录（保留真实后缀）
        shutil.move(str(src), str(out_dir / cat / src.name))
        meta[cat].append({"pdb": pid, "ext": src.suffix,
                          "reason": reason, "L": n_ca,
                          "ligand": list(ligand.keys())[:6]})

    # 各取目标数 + 合并 symlink
    all_dir = out_dir / "all_pdb"
    all_dir.mkdir(exist_ok=True)
    picked = {}
    for cat, n_target in cat_dirs.items():
        picked[cat] = meta[cat][:n_target]
    for cat, lst in picked.items():
        for item in lst:
            src = out_dir / cat / f"{item['pdb']}{item['ext']}"
            lnk = all_dir / f"{item['pdb']}{item['ext']}"
            if not lnk.exists():
                os.symlink(str(src), str(lnk))
        print(f"{cat}: 取 {len(lst)} 个 (候选分类 {len(meta[cat])})", flush=True)

    with open(out_dir / "classification.json", "w") as f:
        json.dump({"picked": picked, "all": meta, "filtered": filtered},
                  f, indent=2, ensure_ascii=False)
    tot = sum(len(v) for v in picked.values())
    print(f"完成：五类合计 {tot} 个（过滤: 长度{filtered['len']} 解析失败{filtered['parse_fail']}）", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampled", type=int, default=15000)
    ap.add_argument("--out", default="data/ligand_train")
    ap.add_argument("--targets", default="rna:1000,dna:1000,small_mol:3000,metal:2500,water:1500")
    ap.add_argument("--parallel", type=int, default=12)
    ap.add_argument("--max_len", type=int, default=300)
    ap.add_argument("--stage", choices=["all", "sample", "download", "classify"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.stage in ("all", "sample"):
        cands = get_candidates(args.sampled)
        (out / "candidates.json").write_text(json.dumps(cands))
        print(f"候选已存 {out}/candidates.json", flush=True)
    else:
        cands = json.load(open(out / "candidates.json"))

    if args.stage in ("all", "download"):
        download_pdbs(cands, out, args.parallel)

    if args.stage in ("all", "classify"):
        classify_and_pick(cands, out, args.targets, args.max_len)


if __name__ == "__main__":
    main()

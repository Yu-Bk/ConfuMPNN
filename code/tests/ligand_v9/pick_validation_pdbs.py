"""v9 泛化验证候选蛋白选择：拉候选 → 下载 → 本地分类 → 防泄漏 → 输出清单。

背景（index/PROJECT_V9_GENERALIZATION_PLAN.md）：验证 v9 编码器在**未见蛋白**上的泛化，
需要 10 个蛋白覆盖小分子/RNA/DNA/金属/长序列 5 类，且不在训练集（4972 复合物）与
验证蛋白（1mbn/4dfr/1fqg/5hvx/3t0f）中。

流程：
  1. RCSB 搜索 API 拉候选（单链蛋白 + 非聚合物配体 + 分辨率≤3.0）
  2. 并行 curl 下载（pdb 优先，404 降级 cif）
  3. 本地 HETATM 解析 → 分类（复用 fetch_ligand_pdbs 规则）+ L + 配体列表
  4. 排除训练集 domain_ids 与验证蛋白前缀
  5. 输出 candidates_for_validation.json（含分类/L/配体）

用法（base 环境，需 requests）：
  python code/tests/ligand_v9/pick_validation_pdbs.py --n 800 --out data/validation_pdbs
输出：
  data/validation_pdbs/{pdb}.pdb|cif    候选 PDB
  data/validation_pdbs/candidates.json   候选清单（分类/L/配体/是否泄漏）
"""
import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from fetch_ligand_pdbs import (  # noqa: E402
    RCSB_QUERY, classify_ligand, parse_ligand_info,
)

EXCLUDE_VALIDATION = {"1mbn", "4dfr", "1fqg", "5hvx", "3t0f"}


def get_candidates(n, resolution=3.0):
    """RCSB 搜索拉候选：单链蛋白 + 非聚合物配体 + 分辨率过滤。"""
    q = json.loads(json.dumps(RCSB_QUERY))  # 深拷贝
    # 放宽分辨率（长序列单链+配体高分辨结构少）
    for node in q["query"]["nodes"]:
        p = node.get("parameters", {})
        if p.get("attribute") == "rcsb_entry_info.resolution_combined":
            p["value"] = resolution
    q["request_options"] = {"paginate": {"start": 0, "rows": 2000}}
    r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"RCSB API 错误: {r.status_code} {r.text[:300]}")
    ids = [h["identifier"] for h in r.json().get("result_set", [])]
    random.seed(42)
    return random.sample(ids, min(n, len(ids)))


def download_pdbs(cands, out_dir, parallel=16):
    out_dir.mkdir(parents=True, exist_ok=True)
    id_file = out_dir / "_ids.txt"
    id_file.write_text("\n".join(cands))
    script = f"""#!/bin/bash
cat {id_file} | xargs -P {parallel} -I{{}} bash -c '
  id="{{}}"; d={out_dir}
  if [ -f "$d/$id.pdb" ] || [ -f "$d/$id.cif" ]; then exit 0; fi
  if curl -sfL --max-time 60 -o "$d/$id.pdb" "https://files.rcsb.org/download/$id.pdb" 2>/dev/null && [ -s "$d/$id.pdb" ]; then exit 0; fi
  rm -f "$d/$id.pdb"
  curl -sfL --max-time 60 -o "$d/$id.cif" "https://files.rcsb.org/download/$id.cif" 2>/dev/null && [ -s "$d/$id.cif" ] || rm -f "$d/$id.cif"
'
"""
    tmp = out_dir / "_download.sh"
    tmp.write_text(script)
    subprocess.run(["bash", str(tmp)])
    n = sum(1 for p in out_dir.glob("*.pdb")) + sum(1 for p in out_dir.glob("*.cif"))
    print(f"下载完成 {n}/{len(cands)}", flush=True)
    return n


def load_train_ids():
    """训练集 domain_ids（防泄漏）。"""
    import numpy as np
    d = np.load("data/ligand_train/labels.npz", allow_pickle=True)
    return {str(x).lower() for x in d["domain_ids"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--out", default="data/validation_pdbs")
    ap.add_argument("--resolution", type=float, default=3.0)
    ap.add_argument("--skip_download", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_ids = load_train_ids()

    cands = get_candidates(args.n, args.resolution)
    print(f"候选 {len(cands)}（RCSB 搜索返回）", flush=True)

    if not args.skip_download:
        download_pdbs(cands, out)

    # 本地解析分类 + 防泄漏
    rows = []
    for pid in cands:
        src = None
        for suf in (".pdb", ".cif"):
            p = out / f"{pid}{suf}"
            if p.exists():
                src = p
                break
        if src is None:
            continue
        try:
            ligand, L = parse_ligand_info(src)
        except Exception:
            continue
        if L <= 0:
            continue
        cat, reason = classify_ligand(ligand)
        base = pid.lower()
        leak = (f"{pid}.{src.suffix[1:]}".lower() in train_ids
                or base in EXCLUDE_VALIDATION)
        rows.append({"pdb": pid, "ext": src.suffix, "L": L,
                     "cat": cat, "reason": reason,
                     "ligand": list(ligand.keys())[:6], "leak": leak})

    # 分类统计
    from collections import Counter
    cnt = Counter(r["cat"] for r in rows)
    leak_cnt = Counter(r["cat"] for r in rows if r["leak"])
    print(f"分类统计（共 {len(rows)} 可解析）: {dict(cnt)}")
    print(f"泄漏/验证蛋白排除: {dict(leak_cnt)}", flush=True)

    # 长序列分桶（L≥400）
    long = [r for r in rows if r["L"] >= 400]
    print(f"长序列候选（L≥400）: {len(long)} 个", flush=True)

    with open(out / "candidates.json", "w") as f:
        json.dump({"rows": rows, "stats": {"by_cat": dict(cnt),
                                           "long": len(long)}},
                  f, indent=2)
    print(f"已写 {out}/candidates.json", flush=True)

    # 打印各类非泄漏候选概览（前 15）
    for cat in ("small_mol", "rna", "dna", "metal"):
        lst = [r for r in rows if r["cat"] == cat and not r["leak"]][:15]
        print(f"\n--- {cat}（非泄漏前15）---")
        for r in lst:
            print(f"  {r['pdb']} L={r['L']:4d} {r['reason'][:40]}")
    print(f"\n--- 长序列（非泄漏，L≥400）---")
    for r in [x for x in long if not x["leak"]][:15]:
        print(f"  {r['pdb']} L={r['L']:4d} cat={r['cat']} {r['reason'][:40]}")


if __name__ == "__main__":
    main()

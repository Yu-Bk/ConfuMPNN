"""阶段 2（v2）：从合并池按配额+类型内 L/q 分层选满 805（轨 B，2026-09-04）。

池：
  /tmp/v14_combined_pool.json  （main 900 + topup long/deepneg 341，已跨池按 id 去重，含 type/coverage）
  /tmp/rna_ext_cov.json        （RNA/DNA 本地，type=RNA/DNA）
只保留 coverage in/boundary。配额：small_mol 621 / metal 85 / nucleotide 37 / RNA-DNA 62。
分层网格：L 箱 [<150,150-250,250-350,>=350] × q 带 [<-20,-20..-15,-15..-8,-8..-3,-3..3,3..8,8..15,>15]，
按训练对应类型占比分配（最大余数法），每格固定 seed 随机抽；不足由同类型余量按 q 近邻补。
输出：/tmp/v14_805_selected.json
"""
import json
import random
from collections import Counter

import numpy as np
from src.differentiable_charge import net_charge

QUOTA = {"small_mol": 621, "metal": 85, "nucleotide": 37, "RNA/DNA": 62}
L_BANDS = [(0, 150), (150, 250), (250, 350), (350, 10 ** 9)]
Q_BANDS = [(-10 ** 9, -20), (-20, -15), (-15, -8), (-8, -3), (-3, 3), (3, 8), (8, 15), (15, 10 ** 9)]
SEED = 2026


def band_of(L, q):
    li = next(i for i, (a, b) in enumerate(L_BANDS) if a <= L < b)
    qi = next(i for i, (a, b) in enumerate(Q_BANDS) if a <= q < b)
    return li, qi


def train_grid(ttype):
    import os
    d = np.load("data/ligand_train/labels_v14_final.npz", allow_pickle=True)
    ids = [str(x) for x in d["domain_ids"]]
    cat = {}
    for did in ids:
        tgt = os.readlink("data/ligand_train/all_pdb/" + did)
        c = tgt.split("/")[1] if tgt.startswith("../") else os.path.dirname(tgt).split("/")[-1]
        cat[did] = {"small_mol": "small_mol", "metal": "metal",
                    "rna": "nucleotide", "dna": "nucleotide", "rna_pdbs": "RNA/DNA"}[c]
    Ls = np.array([len(str(s)) for s, did in zip(d["seqs"], ids) if cat[did] == ttype])
    qs = np.array([net_charge(str(s), 7.4) for s, did in zip(d["seqs"], ids) if cat[did] == ttype])
    cnt = Counter(band_of(L, q) for L, q in zip(Ls, qs))
    return cnt, sum(cnt.values())


def pick_stratified(cands, quota, ttype):
    grid, ntrain = train_grid(ttype)
    cand_by_cell = {}
    for r in cands:
        cand_by_cell.setdefault(band_of(r["L"], r["q"]), []).append(r)
    floors = {}
    rem = []
    for cell, frac in grid.items():
        exact = quota * frac / ntrain
        floors[cell] = int(exact)
        rem.append((exact - int(exact), cell))
    used = sum(floors.values())
    rem.sort(reverse=True)
    for _, cell in rem:
        if used >= quota:
            break
        floors[cell] += 1
        used += 1
    rng = random.Random(SEED)
    chosen = []
    for cell, tgt in floors.items():
        pool = cand_by_cell.get(cell, [])
        rng.shuffle(pool)
        chosen.extend(pool[:tgt])
    if len(chosen) < quota:
        taken_ids = {r["id"] for r in chosen}
        rest = [r for r in cands if r["id"] not in taken_ids]
        rest.sort(key=lambda r: (r["L"], r["q"]))
        for r in rest:
            if len(chosen) >= quota:
                break
            chosen.append(r)
    return chosen, used, len(cands)


def main():
    comb = json.load(open("/tmp/v14_combined_pool.json"))
    rna = json.load(open("/tmp/rna_ext_cov.json"))
    for r in rna:
        r["type"] = "RNA/DNA"
    allc = [r for r in comb if r.get("coverage") != "out"] + \
           [r for r in rna if r.get("coverage") != "out"]
    by_type = {t: [r for r in allc if r["type"] == t] for t in QUOTA}
    selected = []
    report = {}
    for t, quota in QUOTA.items():
        chosen, used, pooln = pick_stratified(by_type[t], quota, t)
        print(f"{t}: pool_ib={pooln} -> selected {len(chosen)}")
        selected.extend(chosen)
        report[t] = {"pool_ib": pooln, "selected": len(chosen)}
    seen = set(); uniq = []
    for r in selected:
        if r["id"] in seen:
            continue
        seen.add(r["id"]); uniq.append(r)
    print("total:", len(uniq))
    json.dump(uniq, open("/tmp/v14_805_selected.json", "w"), indent=1)
    print("by type:", dict(Counter(r["type"] for r in uniq)))
    print("by coverage:", dict(Counter(r["coverage"] for r in uniq)))
    # length summary
    for t in QUOTA:
        sub = [r for r in uniq if r["type"] == t]
        L = [r["L"] for r in sub]
        print(f"  {t}: L med {int(np.median(L))} min {min(L)} max {max(L)} "
              f"<200 {sum(1 for x in L if x<200)} >=350 {sum(1 for x in L if x>=350)} "
              f"q<=-20 {sum(1 for r in sub if r['q']<=-20)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""从各 *_gen_stats.json 聚"蛋白×电荷臂"的 RMSD/TM/pLDDT 矩阵（2026-09-06）。
用法: python aggregate_rmsd_matrix.py <gen_stats.json...> -o out.json
输出: {tag? : {"native":{pdb:rmsd}, ...}} 以文件为列。
"""
import json,sys
def collect(path):
    d=json.load(open(path))
    mat={"arms":{}}
    for arm in ["native","n2","p2","n8","p8"]:
        row={}
        for p,v in d.get("proteins",{}).items():
            a=v.get("mode",{}).get("ligand",{}).get("arms",{}).get(arm) or v.get("arms",{}).get(arm)
            if a: row[p]={"rmsd":a.get("rmsd_median"),"tm":a.get("tm_median"),"plddt":a.get("plddt_median")}
        mat["arms"][arm]=row
    return mat
if __name__=="__main__":
    out=sys.argv[sys.argv.index("-o")+1] if "-o" in sys.argv else "output/rmsd_matrix.json"
    args=[a for a in sys.argv[1:] if not a.startswith("-o") and a != out]
    res={p:collect(p) for p in args}
    json.dump(res,open(out,"w"),indent=1)
    for p in args:
        m=res[p]; n0=m["arms"]["native"]
        print(p,"arms:",{a:len(v) for a,v in m["arms"].items()}, "native 例:", {k:v['rmsd'] for k,v in list(n0.items())[:3]})

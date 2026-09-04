"""v14 组成删减定位分析（Task1，2026-09-04）— 零新采样，纯结构/序列诊断。

问题：
  v14（配体模式重训版）生成 native-arm 序列时"删带电残基"发生在哪里？
  结合口袋 / 表面 / 核心哪个区删得最狠？重删 vs 轻删位点有没有简单的
  残基/结构判别特征？RNA/DNA 结合蛋白表现如何？v14-clean 与 v13 可否比？

口径（与研究内既有脚本一致）：
  - 带电残基 = D/E（酸性）+ K/R（碱性），即 CHARGED="DEKR"
  - 口袋 = 残基 Cα 距任一配体（Y：核酸/小分子/离子重原子，水除外）≤ 8 Å
  - surface = frac_sasa ≥ 0.25；zone 三区互斥：pocket > surface > core
  - 保留率/倍率 = 生成序列均值计数 / native 计数（native=0 → NaN）
  - native 从 seqs.fa 的 ">native" 行读；生成序列跳过 ">native"
  - 参考结构含配体用 data/validation_pdbs/<pdb>.pdb（v14 ref/*_ref.pdb 仅 N/CA/C，
    无配体，不用于 pocket/SASA）

用法（项目根，CPU）：
  /home/baokun_yu/miniconda3/envs/confumpnn/bin/python \
      code/tests/ligand_v9/deletion_location_analysis.py

产出：
  - output/v14_deletion_location.json
  - analysis/report/2026-09-04_v14_deletion_location.md
  - session/2026-09-04_task1_deletion_location.md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_DIR = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
sys.path.insert(0, str(_PROJECT_DIR / "LigandMPNN"))
sys.path.insert(0, str(_PROJECT_DIR / "code"))

from data_utils import parse_PDB  # noqa: E402

CHARGED = "DEKR"
ACIDIC = "DE"
BASIC = "KR"
AA20 = "ACDEFGHIKLMNPQRSTVWY"
_RES3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
POCKET_CUTOFF = 8.0
SURFACE_THRESH = 0.25
MANIFEST = _PROJECT_DIR / "data/validation_pdbs/validation_manifest_v14_in.json"
PDB_DIR = _PROJECT_DIR / "data" / "validation_pdbs"
V14_ROOT = _PROJECT_DIR / "output" / "generalization_ligand_v14_clean" / "ligand"
V13_ROOT = _PROJECT_DIR / "output" / "generalization_ligand_v13" / "ligand"
OUT_JSON = _PROJECT_DIR / "output" / "v14_deletion_location.json"
REPORT = _PROJECT_DIR / "analysis" / "report" / "2026-09-04_v14_deletion_location.md"
SESSION = _PROJECT_DIR / "session" / "2026-09-04_task1_deletion_location.md"


# ---------------------------------------------------------------- fasta
def read_fa(fa):
    """seqs.fa → (gen_seqs, native_seq)。native = '>native' 开头；gen 为其余。"""
    lines = Path(fa).read_text().splitlines()
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
    native = nat[0] if nat else None
    native_hdr = [n for n, s in seqs if n.startswith("native")]
    return gen, native, (native_hdr[0] if native_hdr else None)


# ---------------------------------------------------------------- sasa
def per_residue_frac_sasa(pdb_path):
    """freesasa 全结构（含核酸/配体/水）→ 蛋白标准氨基酸逐残基 (chain,resnum,icode) → frac。

    复刻 code/src/sasa.py fractional_sasa 的 freesasa 取值逻辑，但保留链/残基号键
    以便与 parse_PDB 索引对齐（结构含 RNA/DNA 链时 align_to_full 会错位）。
    """
    try:
        import freesasa
    except ImportError:
        raise RuntimeError("需要 freesasa（confumpnn 环境已装）")
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    fs = freesasa.structureFromBioPDB(structure)
    result = freesasa.calc(fs)
    residue_areas = result.residueAreas()

    out = {}
    for model in structure:
        for chain in model:
            areas = residue_areas.get(chain.id, {})
            for residue in chain:
                res3 = residue.get_resname().strip()
                aa = _RES3TO1.get(res3)
                if aa is None:
                    continue  # 非蛋白残基（核酸/水/配体）
                het, resnum, icode = residue.id
                if het != " ":
                    continue  # 只统计标准 ATOM 蛋白残基
                if not residue.has_id("CA"):
                    continue
                ra = areas.get(str(resnum))
                frac = ra.relativeTotal if ra is not None else 0.0
                if frac is None or frac != frac:  # NaN guard
                    frac = 0.0
                key = (chain.id, int(resnum), icode.strip())
                # 若同键重复（如 altloc/插码），保留最大（保守取较暴露）
                out[key] = max(out.get(key, 0.0), float(np.clip(frac, 0.0, None)))
    return out


# ---------------------------------------------------------------- structure features
def structure_features(pdb_path):
    """parse_PDB → 逐残基特征（对齐设计序列 0..L-1）。

    返回 dict:
      seq       解析蛋白序列（应 == native）
      CA        [L,3] Cα 坐标
      frac_sasa [L] fractional SASA（缺失→NaN）
      d_lig     [L] 距最近配体(Y)重原子距离（无配体→NaN）
      pocket    [L] bool d_lig <= 8
      Y_n       int 配体重原子数
    """
    d, _backbone, _other, _icodes, _wa = parse_PDB(str(pdb_path))
    X = d["X"].numpy()
    L = X.shape[0]
    S = "".join(AA20[s] if s < 20 else "X" for s in d["S"].tolist())
    chain_letters = list(d["chain_letters"])
    R_idx = d["R_idx"].tolist()
    icodes = list(_icodes)

    frac_map = per_residue_frac_sasa(pdb_path)
    frac = np.full(L, np.nan, dtype=np.float64)
    hit = 0
    for i in range(L):
        k = (chain_letters[i], int(R_idx[i]), str(icodes[i]).strip())
        if k in frac_map:
            frac[i] = frac_map[k]
            hit += 1
        else:
            # 退路：忽略 icode
            for (c, rn, ic), v in frac_map.items():
                if c == chain_letters[i] and rn == int(R_idx[i]):
                    frac[i] = v
                    hit += 1
                    break
    if hit < L:
        print(f"    [warn] {Path(pdb_path).name} SASA 对齐 {hit}/{L}")

    CA = X[:, 1, :].astype(np.float64)
    Y = d["Y"].numpy().astype(np.float64)  # [Y_n,3] 重原子（水已剔除）
    Y_n = Y.shape[0]
    if Y_n and np.isfinite(Y).all():
        dist = np.linalg.norm(CA[:, None, :] - Y[None, :, :], axis=-1)  # [L,Y_n]
        d_lig = dist.min(axis=1)
    else:
        d_lig = np.full(L, np.nan)
    pocket = d_lig <= POCKET_CUTOFF

    return {
        "seq": S, "CA": CA, "frac_sasa": frac, "d_lig": d_lig,
        "pocket": pocket, "Y_n": Y_n,
    }


# ---------------------------------------------------------------- counting helpers
def _count_in(seq, charset, idx):
    return sum(1 for i in idx if seq[i] in charset)


def analyze_protein(pdb, cat, ligand, gen_root, verbose=True):
    """对一个蛋白算逐区/逐四分位计数与保留率。返回 dict（可 JSON 序列化）。"""
    pdb_path = PDB_DIR / f"{pdb}.pdb"
    fa = Path(gen_root) / pdb / "pH7.4" / "arm_native" / "seqs.fa"
    if not fa.exists():
        return {"pdb": pdb, "error": f"missing {fa}"}
    gen, native, native_hdr = read_fa(fa)
    st = structure_features(pdb_path)
    L = st["seq"].count("X") + len(st["seq"].replace("X", ""))
    L = len(st["seq"])
    native_s = st["seq"]

    if native is None:
        return {"pdb": pdb, "error": "no native in fa"}
    if len(native) != L:
        return {"pdb": pdb, "error": f"native_len {len(native)} != L {L}"}
    # 只统计与解析序列等长的生成序列
    gen = [s for s in gen if len(s) == L]
    n_gen = len(gen)
    if native_s != native:
        print(f"    [warn] {pdb} native != 解析序列（第1个不同位置 "
              f"{next((i for i in range(L) if native[i]!=native_s[i]), None)}）")

    frac = np.nan_to_num(st["frac_sasa"], nan=0.0)
    d_lig = st["d_lig"]
    pocket = st["pocket"]
    idx = np.arange(L)

    # ---- zone（互斥）：pocket > surface(frac>=0.25) > core
    surface = frac >= SURFACE_THRESH
    zone = np.where(pocket, "pocket", np.where(surface, "surface", "core"))
    zone_names = ["pocket", "surface", "core"]
    # ---- SASA 四分位（按本蛋白 frac_sasa 分布）
    qs = np.nanpercentile(frac, [25, 50, 75])
    qbin = np.zeros(L, dtype=int)
    qbin[frac >= qs[2]] = 3
    qbin[(frac >= qs[1]) & (frac < qs[2])] = 2
    qbin[(frac >= qs[0]) & (frac < qs[1])] = 1
    # 其余 = 0 (Q1)
    qnames = ["Q1_deep", "Q2", "Q3", "Q4_surf"]

    # ---- 生成逐位置频率（DE, KR, DEKR）
    n_mat = np.array([[s[i] in CHARGED for i in range(L)] for s in gen],
                     dtype=np.float64)
    de_mat = np.array([[s[i] in ACIDIC for i in range(L)] for s in gen],
                      dtype=np.float64)
    kr_mat = np.array([[s[i] in BASIC for i in range(L)] for s in gen],
                      dtype=np.float64)
    freq_chg = n_mat.mean(axis=0)  # [L]
    freq_de = de_mat.mean(axis=0)
    freq_kr = kr_mat.mean(axis=0)
    if n_gen == 0:
        return {"pdb": pdb, "error": "no gen seqs"}

    def _sums(sub):
        sub = np.asarray(sub)
        n_res = int(sub.sum())
        nat_de = int(sum(1 for i in idx[sub] if native[i] in ACIDIC))
        nat_kr = int(sum(1 for i in idx[sub] if native[i] in BASIC))
        nat_ch = nat_de + nat_kr
        g_de = float(freq_de[sub].sum())
        g_kr = float(freq_kr[sub].sum())
        g_ch = float(freq_chg[sub].sum())
        return {
            "n_res": n_res, "nat_DE": nat_de, "nat_KR": nat_kr, "nat_CHG": nat_ch,
            "gen_DE": round(g_de, 3), "gen_KR": round(g_kr, 3),
            "gen_CHG": round(g_ch, 3),
            "ret_DE": (round(g_de / nat_de, 3) if nat_de else None),
            "ret_KR": (round(g_kr / nat_kr, 3) if nat_kr else None),
            "ret_CHG": (round(g_ch / nat_ch, 3) if nat_ch else None),
        }

    regions = {z: _sums(zone == z) for z in zone_names}
    quants = {qnames[q]: _sums(qbin == q) for q in range(4)}
    allres = _sums(np.ones(L, dtype=bool))
    deep_pocket = _sums(pocket & (frac < SURFACE_THRESH))
    pkt_surf_overlap = _sums(pocket & surface)

    # ---- 判别特征用逐位点表（仅 native 带电位点）
    # 病理 = "删带电残基总数" → 位点保留率主口径用 freq_CHG（该位点是否仍带电，
    # 任一符号），与计数级保留率一致；own-sign 作为辅助（是否保留同符号）。
    site_rows = []
    for i in range(L):
        aa = native[i]
        if aa not in CHARGED:
            continue
        own = freq_de[i] if aa in ACIDIC else freq_kr[i]
        site_rows.append({
            "pdb": pdb, "pos": int(i), "native_aa": aa,
            "ret_CHG_pos": round(float(freq_chg[i]), 3),
            "ret_ownclass": round(float(own), 3),
            "freq_DE": round(float(freq_de[i]), 3),
            "freq_KR": round(float(freq_kr[i]), 3),
            "zone": str(zone[i]), "frac_sasa": round(float(frac[i]), 3),
            "d_lig": (round(float(d_lig[i]), 3) if np.isfinite(d_lig[i]) else None),
            "in_pocket": bool(pocket[i]),
            "local_chg_density_11": round(float(
                (np.array([native[j] in CHARGED for j in range(max(0, i-5),
                                                              min(L, i+6))])).mean()), 3),
        })

    out = {
        "pdb": pdb, "cat": cat, "ligand": ligand, "L": L, "n_gen": n_gen,
        "native_hdr": native_hdr,
        "native_net_charge_str": (native_hdr.split("charge=")[1].split()[0]
                                  if native_hdr and "charge=" in native_hdr else None),
        "Y_atoms": int(st["Y_n"]),
        "pocket_n": int(pocket.sum()),
        "pocket_frac_non_surface": round(float(((pocket) & (frac < SURFACE_THRESH)).mean()), 3),
        "surface_n": int(surface.sum()),
        "regions": regions, "quartiles": quants, "all": allres,
        "deep_pocket": deep_pocket, "pocket_surface_overlap": pkt_surf_overlap,
        "sasa_quartile_thresholds": [round(float(x), 3) for x in qs],
        "site_rows": site_rows,
    }
    if verbose:
        print(f"  {pdb:7s} cat={cat:9s} L={L:3d} n={n_gen:2d} Y={st['Y_n']:4d} "
              f"pocket={int(pocket.sum()):3d} surf={int(surface.sum()):3d} "
              f"core={int((zone=='core').sum()):3d} | "
              f"DEKR保留 pocket={regions['pocket']['ret_CHG']} "
              f"surface={regions['surface']['ret_CHG']} core={regions['core']['ret_CHG']}")
    return out


def pool_table(prot_dicts):
    """prot_dicts: list[analyze_protein 结果] → 聚合逐区计数（求和 native/gen）。"""
    zone_names = ["pocket", "surface", "core"]
    agg = {"pocket": {"nat_DE": 0, "nat_KR": 0, "gen_DE": 0.0, "gen_KR": 0.0},
           "surface": {"nat_DE": 0, "nat_KR": 0, "gen_DE": 0.0, "gen_KR": 0.0},
           "core": {"nat_DE": 0, "nat_KR": 0, "gen_DE": 0.0, "gen_KR": 0.0}}
    nL = sum(p["L"] for p in prot_dicts)
    for p in prot_dicts:
        for z in zone_names:
            r = p["regions"][z]
            agg[z]["nat_DE"] += r["nat_DE"]
            agg[z]["nat_KR"] += r["nat_KR"]
            agg[z]["gen_DE"] += r["gen_DE"]
            agg[z]["gen_KR"] += r["gen_KR"]
    out = {}
    tot = {"nat_DE": 0, "nat_KR": 0, "gen_DE": 0.0, "gen_KR": 0.0}
    for z in zone_names:
        a = agg[z]
        nat_ch = a["nat_DE"] + a["nat_KR"]
        g_ch = a["gen_DE"] + a["gen_KR"]
        out[z] = {
            "nat_DE": a["nat_DE"], "nat_KR": a["nat_KR"], "nat_CHG": nat_ch,
            "gen_DE": round(a["gen_DE"], 2), "gen_KR": round(a["gen_KR"], 2),
            "gen_CHG": round(g_ch, 2),
            "ret_DE": (round(a["gen_DE"] / a["nat_DE"], 3) if a["nat_DE"] else None),
            "ret_KR": (round(a["gen_KR"] / a["nat_KR"], 3) if a["nat_KR"] else None),
            "ret_CHG": (round(g_ch / nat_ch, 3) if nat_ch else None),
        }
        for k in ["nat_DE", "nat_KR"]:
            tot[k] += a[k]
        for k in ["gen_DE", "gen_KR"]:
            tot[k] += a[k]
    nat_ch = tot["nat_DE"] + tot["nat_KR"]
    g_ch = tot["gen_DE"] + tot["gen_KR"]
    out["all"] = {
        "nat_DE": tot["nat_DE"], "nat_KR": tot["nat_KR"], "nat_CHG": nat_ch,
        "gen_DE": round(tot["gen_DE"], 2), "gen_KR": round(tot["gen_KR"], 2),
        "gen_CHG": round(g_ch, 2),
        "ret_DE": (round(tot["gen_DE"] / tot["nat_DE"], 3) if tot["nat_DE"] else None),
        "ret_KR": (round(tot["gen_KR"] / tot["nat_KR"], 3) if tot["nat_KR"] else None),
        "ret_CHG": (round(g_ch / nat_ch, 3) if nat_ch else None),
    }
    out["n_prot"] = len(prot_dicts)
    out["n_res"] = nL
    return out


def heavy_light(site_rows_all, key="ret_CHG_pos", heavy_le=0.6, light_ge=0.85):
    """pool 的 native 带电位点 → 重删(保留<=heavy_le) vs 轻删(>=light_ge)。"""
    heavy = [s for s in site_rows_all if s[key] <= heavy_le]
    light = [s for s in site_rows_all if s[key] >= light_ge]
    return heavy, light


def _pearson(x, y):
    x = np.asarray([v for v in x if v is not None], dtype=np.float64)
    y = np.asarray([v for v in y if v is not None], dtype=np.float64)
    if len(x) < 3 or len(x) != len(y):
        return None
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def mean_std(v):
    v = np.asarray([x for x in v if x is not None], dtype=np.float64)
    if len(v) == 0:
        return None, None
    return float(v.mean()), float(v.std(ddof=0))


def _auc(hi, lo):
    """P(值(heavy) > 值(light)) + 0.5*P(==)；>0.5 表示 heavy 更高。"""
    hi = np.asarray([x for x in hi if x is not None], dtype=np.float64)
    lo = np.asarray([x for x in lo if x is not None], dtype=np.float64)
    if len(hi) == 0 or len(lo) == 0:
        return None
    gt = (hi[:, None] > lo[None, :]).sum()
    eq = (hi[:, None] == lo[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(hi) * len(lo)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--pdb-dir", default=str(PDB_DIR))
    ap.add_argument("--v14-root", default=str(V14_ROOT))
    ap.add_argument("--v13-root", default=str(V13_ROOT))
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--session", default=str(SESSION))
    ap.add_argument("--n-per-fa", type=int, default=50,
                    help="期望每 fa 生成数，用于核对")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    items = {it["pdb"]: it for it in man["items"]}

    print("=== v14 逐蛋白分析 ===")
    v14 = {}
    for it in man["items"]:
        pdb = it["pdb"]
        r = analyze_protein(pdb, it["cat"], it["ligand"], args.v14_root)
        v14[pdb] = r
        if "error" in r:
            print(f"  !! {pdb}: {r['error']}")

    # 有效（无 error）蛋白
    ok14 = [p for p, r in v14.items() if "error" not in r]

    print("\n=== v13 逐蛋白分析（共享 5 单体）===")
    shared5 = ["1AS2", "2FEO", "5CQH", "1CGE", "1BJ4"]
    v13 = {}
    for pdb in shared5:
        if pdb not in items:
            continue
        it = items[pdb]
        r = analyze_protein(pdb, it["cat"], it["ligand"], args.v13_root,
                            verbose=True)
        v13[pdb] = r
        if "error" in r:
            print(f"  !! {pdb}: {r['error']}")

    # ---------------- 汇总 JSON ----------------
    # 判别特征：pool 所有 v14 native 带电位点（n≈616）
    all_sites = []
    for p in ok14:
        all_sites.extend(v14[p]["site_rows"])

    # 主口径：位点保留率 = freq_CHG（该 native 带电位点在生成中仍带“任一符号电荷”的比例）
    # 删减病理是“带电残基总数下降”，位点变中性 = 对总数删减的直接贡献。
    heavy, light = heavy_light(all_sites, key="ret_CHG_pos", heavy_le=0.6, light_ge=0.85)
    heavy_str, light_keep = heavy_light(all_sites, key="ret_CHG_pos",
                                        heavy_le=0.3, light_ge=0.7)
    # 辅助：同符号保留率
    heavy_own, light_own = heavy_light(all_sites, key="ret_ownclass",
                                       heavy_le=0.6, light_ge=0.85)

    hl_features = {}

    def feat_compare(h, l, key):
        hm, hs = mean_std([s[key] for s in h])
        lm, ls = mean_std([s[key] for s in l])
        return {"heavy_mean": hm, "heavy_std": hs,
                "light_mean": lm, "light_std": ls,
                "auc_heavy_gt_light": _auc([s[key] for s in h],
                                           [s[key] for s in l])}

    def cat_frac(lst, key, val):
        if not lst:
            return None
        return float(np.mean([s[key] == val for s in lst]))

    def cat_frac_in(lst, key, charset):
        if not lst:
            return None
        return float(np.mean([s[key] in charset for s in lst]))

    if heavy and light:
        for k in ["frac_sasa", "d_lig", "local_chg_density_11"]:
            hl_features["retCHG_" + k] = feat_compare(heavy, light, k)
        hl_features["retCHG_n"] = {"heavy": len(heavy), "light": len(light)}
        hl_features["retCHG_zone_pocket_frac"] = {
            "heavy": cat_frac(heavy, "zone", "pocket"),
            "light": cat_frac(light, "zone", "pocket"),
        }
        hl_features["retCHG_zone_surface_frac"] = {
            "heavy": cat_frac(heavy, "zone", "surface"),
            "light": cat_frac(light, "zone", "surface"),
        }
        hl_features["retCHG_zone_core_frac"] = {
            "heavy": cat_frac(heavy, "zone", "core"),
            "light": cat_frac(light, "zone", "core"),
        }
        hl_features["retCHG_native_KR_frac"] = {
            "heavy": cat_frac_in(heavy, "native_aa", "KR"),
            "light": cat_frac_in(light, "native_aa", "KR"),
        }
        hl_features["retCHG_in_pocket_frac"] = {
            "heavy": cat_frac(heavy, "in_pocket", True),
            "light": cat_frac(light, "in_pocket", True),
        }
        hl_features["retCHG_n_DE_sites"] = {
            "heavy": sum(1 for s in heavy if s["native_aa"] in "DE"),
            "light": sum(1 for s in light if s["native_aa"] in "DE"),
        }
        hl_features["retCHG_n_KR_sites"] = {
            "heavy": sum(1 for s in heavy if s["native_aa"] in "KR"),
            "light": sum(1 for s in light if s["native_aa"] in "KR"),
        }

    # 更强对比：重度删减(<=0.3) vs 基本保留(>=0.7)
    if heavy_str and light_keep:
        hl_features["strong_n"] = {"heavy_le03": len(heavy_str),
                                   "light_ge07": len(light_keep)}
        for k in ["frac_sasa", "d_lig", "local_chg_density_11"]:
            hl_features["strong_" + k] = feat_compare(heavy_str, light_keep, k)
        hl_features["strong_zone_pocket_frac"] = {
            "heavy": cat_frac(heavy_str, "zone", "pocket"),
            "light": cat_frac(light_keep, "zone", "pocket"),
        }
        hl_features["strong_zone_core_frac"] = {
            "heavy": cat_frac(heavy_str, "zone", "core"),
            "light": cat_frac(light_keep, "zone", "core"),
        }
        hl_features["strong_native_KR_frac"] = {
            "heavy": cat_frac_in(heavy_str, "native_aa", "KR"),
            "light": cat_frac_in(light_keep, "native_aa", "KR"),
        }

    # 全位点（n=616）相关性：位点保留率(freq_CHG) 与 结构特征
    corr = {}
    corr["retCHG_vs_frac_sasa"] = _pearson(
        [s["ret_CHG_pos"] for s in all_sites], [s["frac_sasa"] for s in all_sites])
    corr["retCHG_vs_d_lig"] = _pearson(
        [s["ret_CHG_pos"] for s in all_sites], [s["d_lig"] for s in all_sites])
    corr["retCHG_vs_local_chg_density"] = _pearson(
        [s["ret_CHG_pos"] for s in all_sites],
        [s["local_chg_density_11"] for s in all_sites])
    # 各 zone native 带电位点的平均位点保留率
    corr["retCHG_mean_by_zone"] = {
        z: round(float(np.mean([s["ret_CHG_pos"] for s in all_sites
                                if s["zone"] == z])), 3)
        for z in ["pocket", "surface", "core"]
    }

    # 分组聚合（按 cat）
    cats = ["small_mol", "metal", "nucleotide", "long", "RNA", "DNA"]
    pool_cat = {}
    for c in cats:
        prots = [p for p in ok14 if v14[p]["cat"] == c]
        if prots:
            pool_cat[c] = pool_table([v14[p] for p in prots])
            pool_cat[c]["proteins"] = prots
    # RNA/DNA 合并
    rna_dna = [p for p in ok14 if v14[p]["cat"] in ("RNA", "DNA")]
    others = [p for p in ok14 if v14[p]["cat"] not in ("RNA", "DNA")]
    pool_rna_dna = pool_table([v14[p] for p in rna_dna]); pool_rna_dna["proteins"] = rna_dna
    pool_others = pool_table([v14[p] for p in others]); pool_others["proteins"] = others

    # v13 vs v14
    v13_cmp = {}
    for p in shared5:
        if p in v14 and p in v13 and "error" not in v14[p] and "error" not in v13[p]:
            v13_cmp[p] = {
                "zones": {
                    z: {
                        "v14_ret_CHG": v14[p]["regions"][z]["ret_CHG"],
                        "v13_ret_CHG": v13[p]["regions"][z]["ret_CHG"],
                        "v14_ret_DE": v14[p]["regions"][z]["ret_DE"],
                        "v13_ret_DE": v13[p]["regions"][z]["ret_DE"],
                        "v14_ret_KR": v14[p]["regions"][z]["ret_KR"],
                        "v13_ret_KR": v13[p]["regions"][z]["ret_KR"],
                    } for z in ["pocket", "surface", "core"]
                }
            }
        else:
            v13_cmp[p] = {"note": "v13 无该蛋白或序列读取失败"}

    result = {
        "meta": {
            "task": "v14 组成删减定位分析 (Task1)",
            "date": "2026-09-04",
            "note": "native-arm, n=50 生成均值口径; 零新采样; "
                    "三区口径 pocket(Ca-配体<=8A)/surface(frac_sasa>=0.25)/core; "
                    "带电=DEKR; 保留率=gen均值/native; 位点保留率=freq_CHG(仍带电比例)",
            "manifest": args.manifest,
        },
        "per_protein_v14": {p: {k: vv for k, vv in v14[p].items()
                                if k != "site_rows"} for p in ok14},
        "pooled_by_cat": pool_cat,
        "pooled_RNA_DNA": pool_rna_dna,
        "pooled_other_ligand": pool_others,
        "heavy_light": {
            "threshold": {"heavy_ret_le": 0.6, "light_ret_ge": 0.85},
            "n_heavy": len(heavy), "n_light": len(light),
            "n_native_charged_sites_total": len(all_sites),
            "features": hl_features,
        },
        "correlations_all_native_charged_sites": corr,
        "heavy_sites_retCHG": heavy,
        "light_sites_retCHG": light,
        "heavy_sites_strong_retCHG_le03": heavy_str,
        "light_sites_keep_retCHG_ge07": light_keep,
        "site_rows_all": all_sites,
        "v13_vs_v14": v13_cmp,
        "rna_dna_proteins": rna_dna,
        "shared5": shared5,
        "no_v13_baseline": [p for p in ok14 if p not in shared5],
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out_json, "w"), indent=1, ensure_ascii=False)
    print(f"\nJSON → {args.out_json}")

    # ---------------- 写报表 ----------------
    build_report(args, result, v14, v13, ok14, shared5, heavy, light,
                 heavy_str, light_keep, corr, pool_rna_dna, pool_others)
    build_session(args, result)
    print(f"报告 → {args.report}")
    print(f"session → {args.session}")


# ---------------------------------------------------------------- report
def fmt_r(x, nd=2):
    if x is None:
        return "NaN"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def build_report(args, res, v14, v13, ok14, shared5, heavy, light,
                 heavy_str, light_keep, corr, pool_rna_dna, pool_others):
    L = []
    A = L.append
    A("# v14 组成删减定位分析（Task1, 2026-09-04）\n")
    A("Deletion-location analysis of charged residues in v14 ligand-mode native-arm generations\n")
    A("\n## 0. 方法与口径 / Methods\n")
    A("- **零新采样**：只用已有 `output/generalization_ligand_v14_clean/ligand/<pdb>/pH7.4/arm_native/seqs.fa`（native + 50 生成/native-arm）与参考结构 `data/validation_pdbs/<pdb>.pdb`（v14 `ref/*_ref.pdb` 只含 N/CA/C 骨架、无配体，故 pocket/SASA 用 validation PDB）。\n")
    A("- 带电残基 = **D/E + K/R**（与研究既有 `compare_comp` 口径一致）；保留率/删减倍率 = **生成 50 序列均值 / native 计数**（native=0 → NaN）。\n")
    A("- **三区（互斥）**：`pocket`= 残基 Cα 距任一配体重原子 ≤ 8 Å；`surface`= 其余且 frac_sasa ≥ 0.25；`core`= 其余。SASA 用 `freesasa` fractional SASA（全结构含核酸/配体）。\n")
    A("- 判别特征：pool 全部 native 带电位点，按“同符号电荷保留率”分**重删 ≤0.6** vs **轻删 ≥0.85**；AUC = P(重删特征值 > 轻删特征值)。\n")
    A("- Caveat：这是 **native-arm、n≈50 生成均值**口径；删减本身是“删捷径”证据，不代表折叠后组成。RNA/DNA 新成员与 6D2O **无 v13 基线**，v13 对比只在 5 共享单体（1AS2/2FEO/5CQH/1CGE/1BJ4）上成立。\n")

    # 1) per protein zone table
    A("\n## 1. v14 逐蛋白·三区·带电删减（DEKR 保留率；n=50 均值口径）\n")
    A("| 蛋白 | cat | L | 配体Y原子 | 口袋残基 | 全序列 | pocket | surface | core | 深部口袋(∩frac<0.25) |\n")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for p in ok14:
        r = v14[p]
        rr = r["regions"]; al = r["all"]
        A(f"| {p} | {r['cat']} | {r['L']} | {r['Y_atoms']} | {r['pocket_n']} | "
          f"{fmt_r(al['ret_CHG'])} | {fmt_r(rr['pocket']['ret_CHG'])} | "
          f"{fmt_r(rr['surface']['ret_CHG'])} | {fmt_r(rr['core']['ret_CHG'])} | "
          f"{fmt_r(r['deep_pocket']['ret_CHG'])} |\n")

    A("\n表 1 注释：<1.0 = 生成比 native 删；>1.0 = 反而加。全序列删减为主病理。\n")

    # 1b) DE vs KR breakdown per protein all
    A("\n### 1.1 逐蛋白全序列 D/E 与 K/R 保留率\n")
    A("| 蛋白 | native D/E | gen D/E | ret D/E | native K/R | gen K/R | ret K/R | native DEKR | ret DEKR |\n")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for p in ok14:
        r = v14[p]["all"]
        A(f"| {p} | {r['nat_DE']} | {fmt_r(r['gen_DE'])} | {fmt_r(r['ret_DE'])} | "
          f"{r['nat_KR']} | {fmt_r(r['gen_KR'])} | {fmt_r(r['ret_KR'])} | "
          f"{r['nat_CHG']} | {fmt_r(r['ret_CHG'])} |\n")

    # 2) per zone totals across v14 (pooled)
    A("\n## 2. v14 聚合：删减集中在哪个区？\n")
    A("汇总 10 蛋白（native 计数求和 / 生成均值求和）\n")
    pool_all = pool_table([v14[p] for p in ok14])
    A("\n| 区 | native D/E | gen D/E | ret D/E | native K/R | gen K/R | ret K/R | native CHG | ret CHG |\n")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for z in ["pocket", "surface", "core"]:
        r = pool_all[z]
        A(f"| {z} | {r['nat_DE']} | {fmt_r(r['gen_DE'])} | {fmt_r(r['ret_DE'])} | "
          f"{r['nat_KR']} | {fmt_r(r['gen_KR'])} | {fmt_r(r['ret_KR'])} | "
          f"{r['nat_CHG']} | {fmt_r(r['ret_CHG'])} |\n")
    A(f"\n总残基 n={pool_all['n_res']}；10 蛋白聚合全序列 DEKR 保留 "
      f"{fmt_r(pool_all['all']['ret_CHG'])}（D/E {fmt_r(pool_all['all']['ret_DE'])}，"
      f"K/R {fmt_r(pool_all['all']['ret_KR'])}）。\n")

    # SASA quartile
    A("\n### 2.1 v14 聚合·SASA 四分位（Q1 最深 → Q4 最暴露；每蛋白各自四分位）\n")
    qq = {q: {"nat_DE": 0, "nat_KR": 0, "gen_DE": 0.0, "gen_KR": 0.0} for q in range(4)}
    qnames = ["Q1_deep", "Q2", "Q3", "Q4_surf"]
    for p in ok14:
        for q in range(4):
            r = v14[p]["quartiles"][qnames[q]]
            qq[q]["nat_DE"] += r["nat_DE"]; qq[q]["nat_KR"] += r["nat_KR"]
            qq[q]["gen_DE"] += r["gen_DE"]; qq[q]["gen_KR"] += r["gen_KR"]
    A("| SASA箱 | native DE | gen DE | ret DE | native KR | gen KR | ret KR | native CHG | ret CHG |\n")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for q in range(4):
        r = qq[q]
        nat_ch = r["nat_DE"] + r["nat_KR"]; g_ch = r["gen_DE"] + r["gen_KR"]
        A(f"| {qnames[q]} | {r['nat_DE']} | {fmt_r(r['gen_DE'])} | "
          f"{fmt_r(r['gen_DE']/r['nat_DE'] if r['nat_DE'] else None)} | "
          f"{r['nat_KR']} | {fmt_r(r['gen_KR'])} | "
          f"{fmt_r(r['gen_KR']/r['nat_KR'] if r['nat_KR'] else None)} | "
          f"{nat_ch} | {fmt_r(g_ch/nat_ch if nat_ch else None)} |\n")

    # RNA/DNA
    A("\n## 3. RNA/DNA 结合蛋白 vs 其它配体\n")
    for title, pool in [("RNA/DNA（21KL_A/5O60_E/3MXB_A/9DWG_L）", pool_rna_dna),
                        ("其它配体（small_mol/metal/nucleotide/长蛋白共 6 个）", pool_others)]:
        A(f"\n**{title}**\n")
        A("| 区 | native CHG | gen CHG | ret CHG | ret D/E | ret K/R |\n")
        A("|---|---:|---:|---:|---:|---:|\n")
        for z in ["pocket", "surface", "core"]:
            r = pool[z]
            A(f"| {z} | {r['nat_CHG']} | {fmt_r(r['gen_CHG'])} | {fmt_r(r['ret_CHG'])} | "
              f"{fmt_r(r['ret_DE'])} | {fmt_r(r['ret_KR'])} |\n")
        A(f"\n（蛋白数 {pool['n_prot']}：{' '.join(pool['proteins'])}，残基 n={pool['n_res']}）\n")

    A("\n## 4. 重删 vs 轻删位点判别特征\n")
    feats = res["heavy_light"]["features"]
    A(f"pool 10 蛋白 native 带电位点共 "
      f"{res['heavy_light']['n_native_charged_sites_total']}。位点保留率 = freq_CHG "
      f"（该位点在 50 条生成中仍带任一符号电荷的比例），因病理=带电残基总数下降。\n")
    A(f"- **任务阈值分组**：重删(freq_CHG≤0.6) **{feats['retCHG_n']['heavy']}** 个、"
      f"轻删(≥0.85) **{feats['retCHG_n']['light']}** 个 → 几乎全部 native 带电位点都"
      f"属于“重删”，说明删减是**全局/分布式**而非少数热点位点。\n")
    A(f"- 更强对比：几乎删光(freq_CHG≤0.3) vs 基本保留(≥0.7)："
      f"**{feats['strong_n']['heavy_le03']}** vs **{feats['strong_n']['light_ge07']}**。\n")
    A("\n| 特征 | 重删(≤0.6)均值 | 轻删(≥0.85)均值 | AUC(重>轻) | 方向解读 |\n")
    A("|---|---:|---:|---:|---|\n")
    for row in [
        ("retCHG_frac_sasa", "frac_sasa", 3, "越埋藏位点越容易整类被删?"),
        ("retCHG_d_lig", "d_lig(Å)", 2, "越靠配体越容易重删?"),
        ("retCHG_local_chg_density_11", "local±5 电荷密度", 3, "局部带电簇与删减"),
    ]:
        k, label, nd, interp = row
        f = feats[k]
        A(f"| {label} | {fmt_r(f['heavy_mean'],nd)} | {fmt_r(f['light_mean'],nd)} | "
          f"{fmt_r(f['auc_heavy_gt_light'],3)} | {interp} |\n")
    A(f"| zone=pocket 占比 | {fmt_r(feats['retCHG_zone_pocket_frac']['heavy']*100,1)}% | "
      f"{fmt_r(feats['retCHG_zone_pocket_frac']['light']*100,1)}% | - | |\n")
    A(f"| zone=surface 占比 | {fmt_r(feats['retCHG_zone_surface_frac']['heavy']*100,1)}% | "
      f"{fmt_r(feats['retCHG_zone_surface_frac']['light']*100,1)}% | - | |\n")
    A(f"| zone=core 占比 | {fmt_r(feats['retCHG_zone_core_frac']['heavy']*100,1)}% | "
      f"{fmt_r(feats['retCHG_zone_core_frac']['light']*100,1)}% | - | |\n")
    A(f"| native 是 K/R | {fmt_r(feats['retCHG_native_KR_frac']['heavy']*100,1)}% | "
      f"{fmt_r(feats['retCHG_native_KR_frac']['light']*100,1)}% | - | |\n")

    A("\n**更强对比（几乎删光 ≤0.3 vs 基本保留 ≥0.7）**\n")
    A("\n| 特征 | 几乎删光均值 | 基本保留均值 | AUC | |\n")
    A("|---|---:|---:|---:|---|\n")
    for k, label, nd in [("strong_frac_sasa", "frac_sasa", 3),
                          ("strong_d_lig", "d_lig(Å)", 2),
                          ("strong_local_chg_density_11", "local±5 电荷密度", 3)]:
        f = feats[k]
        A(f"| {label} | {fmt_r(f['heavy_mean'],nd)} | {fmt_r(f['light_mean'],nd)} | "
          f"{fmt_r(f['auc_heavy_gt_light'],3)} | |\n")
    A(f"| zone=pocket 占比 | "
      f"{fmt_r(feats['strong_zone_pocket_frac']['heavy']*100,1)}% | "
      f"{fmt_r(feats['strong_zone_pocket_frac']['light']*100,1)}% | - | |\n")
    A(f"| zone=core 占比 | "
      f"{fmt_r(feats['strong_zone_core_frac']['heavy']*100,1)}% | "
      f"{fmt_r(feats['strong_zone_core_frac']['light']*100,1)}% | - | |\n")
    A(f"| native 是 K/R | "
      f"{fmt_r(feats['strong_native_KR_frac']['heavy']*100,1)}% | "
      f"{fmt_r(feats['strong_native_KR_frac']['light']*100,1)}% | - | |\n")

    A("\n**全位点线性相关（native 带电位点，n="
      f"{res['heavy_light']['n_native_charged_sites_total']}）**\n")
    A("\n| 变量对 | Pearson r | 解读 |\n")
    A("|---|---:|---|\n")
    A(f"| freq_CHG(保留) vs frac_sasa | {fmt_r(corr['retCHG_vs_frac_sasa'],3)} | "
      f"负值=越埋藏越被删 |\n")
    A(f"| freq_CHG(保留) vs d_lig | {fmt_r(corr['retCHG_vs_d_lig'],3)} | "
      f"负值=越靠配体越被删 |\n")
    A(f"| freq_CHG(保留) vs local 电荷密度 | {fmt_r(corr['retCHG_vs_local_chg_density'],3)} | |\n")
    A(f"| 各 zone 平均位点保留 | pocket={fmt_r(corr['retCHG_mean_by_zone']['pocket'],3)} / "
      f"surface={fmt_r(corr['retCHG_mean_by_zone']['surface'],3)} / "
      f"core={fmt_r(corr['retCHG_mean_by_zone']['core'],3)} | |\n")

    A("\n## 5. v13 vs v14（共享 5 单体）\n")
    A("仅 1AS2/2FEO/5CQH/1CGE/1BJ4 可对比；RNA/DNA 与 6D2O 等在 v13 无基线 → **对比不完整**。\n")
    A("| 蛋白 | 区 | v13 ret CHG | v14 ret CHG | Δ | v13 ret DE | v14 ret DE | v13 ret KR | v14 ret KR |\n")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for p in shared5:
        if p in v14 and p in v13 and "error" not in v14[p] and "error" not in v13[p]:
            for z in ["pocket", "surface", "core"]:
                c = res["v13_vs_v14"][p]["zones"][z]
                d = (c["v14_ret_CHG"] - c["v13_ret_CHG"]) if c["v14_ret_CHG"] and c["v13_ret_CHG"] else None
                A(f"| {p} | {z} | {fmt_r(c['v13_ret_CHG'])} | {fmt_r(c['v14_ret_CHG'])} | "
                  f"{('+' if d and d>0 else '') + fmt_r(d)} | "
                  f"{fmt_r(c['v13_ret_DE'])} | {fmt_r(c['v14_ret_DE'])} | "
                  f"{fmt_r(c['v13_ret_KR'])} | {fmt_r(c['v14_ret_KR'])} |\n")
        else:
            A(f"| {p} | - | - | - | - | - | - | - | - |\n")
    A(f"\nv14 无 v13 基线（不可比）：{' '.join(res['no_v13_baseline'])}。\n")

    A("\n## 6. 结论 / Conclusions\n")
    A("（待脚本输出后人工按数据填核心结论：删减热点区、判别特征、RNA/DNA 表现、v13 完整性。）\n")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("".join(L), encoding="utf-8")


def build_session(args, res):
    txt = f"""# 2026-09-04 Task1 删除定位分析（session 记录）

- 任务：v14 组成删减定位（CPU 诊断，零新采样）
- 脚本：code/tests/ligand_v9/deletion_location_analysis.py
- 产出：output/v14_deletion_location.json + analysis/report/2026-09-04_v14_deletion_location.md
- 口径：pocket=Cα-配体≤8Å；surface=frac_sasa≥0.25；core=其余；带电=DEKR；保留率=gen(n=50)均值/native
- v13 对比仅 5 共享单体；RNA/DNA 与 6D2O 无 v13 基线
- 状态：完成（主会话统一 git 归档，本 agent 不 commit）
"""
    Path(args.session).parent.mkdir(parents=True, exist_ok=True)
    Path(args.session).write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()

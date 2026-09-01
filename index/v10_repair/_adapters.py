"""配体模式（ligand_mpnn）诊断/校准的适配工具集（2026-09-01 三坑修复）。

背景：v12.2 配体迁移诊断（2026-09-01）连续踩了三个环境/库层面的坑，都会让
配体模式脚本直接崩溃。这里统一修复，供 v10_diag_response_curve.py /
build_calibration.py / build_calibration_small.py 等 v10_repair 下脚本复用。
这些修复只影响配体诊断/校准路径，不影响 protein_mpnn（mompnn）训练/验证。

三个坑：
1. argparse 负值：`--targets -34,...` 中 `-34` 被 argparse 误认为选项
   → 报 "argument --targets: expected one argument"。修：parse_args 前自动合并为
   `--targets=-34,...`。
2. prody 无扩展名：prody 2.4.1 对无扩展名文件（如 data/cath/S40/dompdb/7pujA01）
   按 mmCIF 解析 → "mmCIF file contained no atoms"（文件实为 PDB 格式）。
   修：无结构后缀时优先尝试同路径 +'.pdb'（CATH 惯例副本目录 dompdb_pdb/）。
3. 配体模式无配体原子：ligand_mpnn 的 featurize 需要配体原子上下文
   （use_atom_context=True），纯蛋白链（CATH 域）会在 get_nearest_neighbours
   对空配体张量 `L2_AB_nn[:, 0]` 抛 IndexError。修：featurize 包 try/except，
   结构类错误跳过该蛋白并给明确提示（提示选带配体的蛋白），不崩溃。
"""
import sys
from pathlib import Path

from data_utils import featurize  # noqa: E402


def fix_negative_targets(argv):
    """把 '--targets -34,...' 中裸负值自动合并为 '--targets=-34,...'。

    argparse 会把以 '-' 开头的参数值（如 -34）误认为选项。其他以负值开头的
    参数（--decouple_abs_lo）也可仿此扩展，当前仅 --targets 需要。
    用法：ap.parse_args(fix_negative_targets(sys.argv[1:]))
    """
    out = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--targets" and i + 1 < len(argv):
            nxt = argv[i + 1]
            if nxt.startswith("-") and not nxt.startswith("--"):
                out.append("--targets=" + nxt)
                i += 2
                continue
        out.append(a)
        i += 1
    return out


def resolve_pdb_path(path):
    """返回 prody 能正确解析的结构路径；无扩展名时优先找 .pdb 副本。

    prody 2.4.1 对无扩展名文件按 mmCIF 解析失败（'mmCIF file contained no atoms'）。
    带明确后缀（.pdb/.ent/.cif/.mmcif）的文件原样返回；无后缀时依次尝试：
    1) 同路径 +'.pdb'；2) CATH 惯例副本目录（dompdb/<name> → dompdb_pdb/<name>.pdb）。
    都找不到则原样返回（让下游 parse 自行报错，信息更准）。
    """
    p = Path(path)
    if p.suffix.lower() not in (".pdb", ".ent", ".cif", ".mmcif"):
        cand = Path(str(p) + ".pdb")
        if cand.exists():
            return str(cand)
        # CATH 无后缀域（data/cath/S40/dompdb/7pujA01）的 .pdb 副本在
        # 同级 dompdb_pdb/ 目录（data/cath/S40/dompdb_pdb/7pujA01.pdb）
        if p.parent.name == "dompdb":
            cand2 = p.parent.with_name("dompdb_pdb") / (p.name + ".pdb")
            if cand2.exists():
                return str(cand2)
    return str(p)


def safe_featurize(protein_dict, backbone_type, name, feats):
    """featurize 的容错包装：结构类错误（IndexError/KeyError）跳过并提示，不崩溃。

    配体模式对无配体蛋白（纯蛋白链）会在 get_nearest_neighbours 对空配体张量
    `L2_AB_nn[:, 0]` 抛 IndexError。捕获后打印明确提示（配体模式需 HETATM 原子，
    无配体蛋白请换 protein_mpnn 或选带配体的 PDB）并返回 None，调用方 continue。
    只捕获结构数据类错误，不吞 RuntimeError 等真实 bug。
    """
    try:
        return featurize(protein_dict, cutoff_for_score=8.0, **feats)
    except (IndexError, KeyError) as e:
        hint = ("配体模式需蛋白含配体原子（HETATM）；无配体蛋白请用 protein_mpnn "
                "或换成带配体的 PDB") if backbone_type == "ligand_mpnn" else ""
        print(f"  !! {name} featurize 失败（{backbone_type} 模式）：{e}。{hint}",
              flush=True)
        return None

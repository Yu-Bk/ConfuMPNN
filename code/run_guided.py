"""一键运行引导采样（Phase 1 主入口）。

把整条管线串起来：
    PDB → 骨架/配体上下文 → 结构感知过滤器 + 动态电荷前瞻 → 引导采样
    → 生成 N 条候选序列 → 计算每条净电荷/pI → 输出统计

用法（在 code/ 目录下）：
    conda activate confumpnn
    python run_guided.py --pdb input/1BC8.pdb --pH 7.4 [--target_charge -2.0]
                         [--preset default] [--num_samples 10]
    # 默认生成器 = MoMPNN（多目标 DPO 微调版，可溶/热稳更优，E1b 16/16 全优）
    # 显式回退原版 LigandMPNN（含配体上下文）：
    python run_guided.py --pdb input/1BC8.pdb --pH 7.4 \
        --weights ../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt

条件注入模式（Phase 3，用微调后的 ConditionEncoder，模型自身 pH 感知）：
    python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
        --cond_encoder output/finetune/condition_encoder_last.pt \
        [--cond_mode conditioned|baseline] [--no_calibration]
    # conditioned ：注入条件向量（默认），无 logit bias —— 测模型学到的 pH 感知
    # baseline   ：加载编码器但不注入（等价 Phase 1 诚实边界对照）
    # 电荷校准（默认关）：configs/condition_defaults.yaml 的 charge_calibration.enabled=false。
    #   过冲已由训练侧 charge_temp=0.5 根治（v9 起），推理侧校准不再需要。
    #   如需旧式线性校准，可在 yaml 把 enabled 置 true（target_eff=(desired-offset)/gain)；
    #   --no_calibration 强制关闭。

    # pH-only 自动补全（v3 方案 P2/D1/A9）：不传 --target_charge 时，默认自动补全
    #   target = native 序列在 pH 下的净电荷（"保持 native 电荷行为"，落在训练分布内），
    #   不再走 flag=0（训练恒 flag=1，推理 flag=0 从未见过 → 行为不可预测）。
    #   --no_auto_target_charge 关闭自动补全，回到旧 flag=0 语义（A9 对照）。

日志建议重定向到 code/log/，输出写入 code/output/。
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

# ---- 路径设置：code/ 与 LigandMPNN/ ----
_CODE_DIR = Path(__file__).resolve().parent
_LIG_DIR = _CODE_DIR.parent / "LigandMPNN"
for p in [str(_CODE_DIR), str(_LIG_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 默认生成器 = MoMPNN（多目标 DPO 微调版，ProtAlign ICLR 2026）
# E4 决策：E1b 验证 MoMPNN 在 pLDDT/TM/%sol/Tm 四指标 × 4 PDB 全部占优（16/16），
# 设为默认；原版 LigandMPNN（含配体上下文）通过 --weights 显式回退。
_DEFAULT_WEIGHTS = (
    _CODE_DIR.parent / "MoMPNN" / "mompnn_paper_checkpoints"
    / "mompnn_temberture_tm_esm_6_4_4_b01.ckpt"
)

import yaml  # noqa: E402

from data_utils import featurize, parse_PDB, restype_int_to_str  # noqa: E402
from model_utils import ProteinMPNN  # noqa: E402
from src.charge_lookahead import make_dynamic_callback  # noqa: E402
from src.condition_embedding import ConditionEncoder, make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from src.guided_sampler import GuidedSampler, extract_calpha_coords  # noqa: E402
from src.isoelectric_point import find_pI  # noqa: E402
from src.structure_aware_filter import StructureAwareFilter, load_preset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="ConfuMPNN Phase 1 引导采样")
    p.add_argument("--pdb", required=True, help="输入 PDB 路径")
    p.add_argument("--pH", type=float, required=True, help="工作环境 pH")
    p.add_argument("--target_charge", type=float, default=None,
                   help="目标净电荷（None=不引导电荷，只做结构过滤）")
    p.add_argument("--preset", default="default",
                   choices=["default", "nucleic_acid_binding", "membrane", "acidic"],
                   help="结构过滤器场景预设")
    p.add_argument("--num_samples", type=int, default=10, help="生成候选序列数")
    p.add_argument("--temperature", type=float, default=0.3, help="采样温度")
    p.add_argument("--strength", type=float, default=0.5, help="电荷引导强度")
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--weights", default=None,
                   help="权重路径（默认 MoMPNN mompnn_temberture_tm_esm_6_4_4_b01.ckpt；"
                        "回退原版 LigandMPNN 用 ligandmpnn_v_32_010_25.pt）")
    p.add_argument("--model_type", default="auto",
                   choices=["auto", "protein_mpnn", "ligand_mpnn"],
                   help="模型类型：auto=按权重自动检测（默认）；protein_mpnn=纯 backbone（如 MoMPNN）；"
                        "ligand_mpnn=配体上下文（原版 LigandMPNN）")
    p.add_argument("--cond_encoder", default=None,
                   help="微调后的 ConditionEncoder 权重路径（Phase 3 条件注入模式）。"
                        "支持 condition_encoder_last.pt（state dict）或 finetune_epochNNN.pt（含配置）")
    p.add_argument("--cond_mode", default="conditioned",
                   choices=["conditioned", "baseline"],
                   help="条件注入模式：conditioned=注入条件向量（默认，测模型 pH 感知）；"
                        "baseline=加载编码器但不注入（等价 Phase 1 诚实边界对照）")
    p.add_argument("--no_calibration", action="store_true",
                   help="关闭电荷校准（默认开：按 condition_defaults.yaml 的 gain/offset 线性校准 "
                        "target_eff=(desired-offset)/gain，抵消条件注入的 ~2.57× 电荷过冲）")
    p.add_argument("--no_auto_target_charge", action="store_true",
                   help="关闭 pH-only 自动补全（v3 D1/A9）。默认：--target_charge 未给出时自动补全 "
                        "target=native_charge@pH（保持 native 电荷行为）；本开关回到旧 flag=0 语义对照")
    p.add_argument("--fixed_residues", default=None,
                   help="固定残基列表，空格分隔（链字母+残基号，如 'A12 C15'）。"
                        "这些位置的氨基酸保持不变，其余位置由模型设计。"
                        "复用 LigandMPNN 原生机制（chain_mask=0 位置在解码时强制保持原氨基酸）。")
    p.add_argument("--out_dir", default=None,
                   help="输出目录（默认 code/output/guided_<pdb>_pH<pH>）")
    return p.parse_args()


def load_model(weights, device, model_type="auto"):
    checkpoint = torch.load(weights, map_location=device)
    # 自动检测：权重里有 atom_context_num（且 >0）说明是 LigandMPNN 配体权重；
    # 没有则是纯 backbone ProteinMPNN（如 MoMPNN）。
    if model_type == "auto":
        model_type = (
            "ligand_mpnn" if checkpoint.get("atom_context_num", 0) > 0
            else "protein_mpnn"
        )
    atom_context_num = (
        0 if model_type == "protein_mpnn" else int(checkpoint.get("atom_context_num", 16))
    )
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=int(checkpoint["num_edges"]),
        device=device,
        atom_context_num=atom_context_num,
        model_type=model_type,
        ligand_mpnn_use_side_chain_context=0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_condition_encoder(path, device):
    """加载微调后的 ConditionEncoder。

    支持两种 checkpoint 格式：
        condition_encoder_last.pt : 纯 state dict（8 keys，含 mean/std buffer）
        finetune_epochNNN.pt      : dict（含 condition_encoder_state + 配置）
    架构参数（hidden_dim/n_tokens/token_dim/μ/σ）优先取 checkpoint，缺失时用
    configs/condition_defaults.yaml。
    """
    with open(_CODE_DIR / "configs" / "condition_defaults.yaml") as f:
        cfg = yaml.safe_load(f)["condition_defaults"]

    ck = torch.load(path, map_location=device)
    if "condition_encoder_state" in ck:
        state = ck["condition_encoder_state"]
        mean = ck.get("mean", cfg["normalization"]["mean"])
        std = ck.get("std", cfg["normalization"]["std"])
        n_tokens = ck.get("n_tokens", cfg["encoder"]["n_tokens"])
        token_dim = ck.get("token_dim", cfg["encoder"]["token_dim"])
        epoch = ck.get("epoch", None)
    else:
        state = ck
        mean = cfg["normalization"]["mean"]
        std = cfg["normalization"]["std"]
        n_tokens = cfg["encoder"]["n_tokens"]
        token_dim = cfg["encoder"]["token_dim"]
        epoch = "last"

    enc = ConditionEncoder(
        cond_dim=cfg["cond_dim"],
        hidden_dim=cfg["encoder"]["hidden_dim"],
        token_dim=token_dim,
        n_tokens=n_tokens,
        mean=mean,
        std=std,
    )
    enc.load_state_dict(state)
    enc.to(device)
    enc.eval()
    print(f"    条件编码器已加载: {Path(path).name}  (epoch={epoch}, n_tokens={n_tokens}, "
          f"token_dim={token_dim})")
    return enc


def seq_to_string(S):
    return "".join(restype_int_to_str[i] for i in S)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型（默认生成器 = MoMPNN；--weights 可覆盖）
    weights = Path(args.weights) if args.weights else _DEFAULT_WEIGHTS
    print(f"[1] 加载模型: {weights.name}  (device={device})")
    model = load_model(weights, device, model_type=args.model_type)
    mt = model.model_type  # 解析后的实际模型类型
    print(f"    解析 model_type = {mt}")

    # 2. 读 PDB + featurize（按模型类型决定是否用配体上下文）
    print(f"[2] 读取 PDB: {args.pdb}")
    protein_dict, _, _, icodes, _ = parse_PDB(args.pdb)
    protein_dict["chain_mask"] = torch.ones(
        protein_dict["X"].shape[0], dtype=torch.int32  # 默认设计全部残基
    )
    # 位点固定（LigandMPNN 原生机制：chain_mask=0 的位置解码时强制保持原氨基酸，
    # 见 guided_sampler.py 的 S_t = S_t·chain_mask_t + S_true·(1-chain_mask_t)）
    fixed_positions = []
    if args.fixed_residues:
        R_idx = list(protein_dict["R_idx"].cpu().numpy())
        chain_letters = list(protein_dict["chain_letters"])
        encoded = [
            str(chain_letters[i]) + str(R_idx[i]) + icodes[i]
            for i in range(len(R_idx))
        ]
        fixed_set = set(args.fixed_residues.split())
        for i, name in enumerate(encoded):
            if name in fixed_set:
                protein_dict["chain_mask"][i] = 0
                fixed_positions.append(name)
        print(f"    固定残基 {len(fixed_positions)} 个: {fixed_positions}")
    use_atom_context = (mt == "ligand_mpnn")
    feature_dict = featurize(
        protein_dict, cutoff_for_score=8.0,
        use_atom_context=use_atom_context,
        number_of_ligand_atoms=(16 if use_atom_context else 0),
        model_type=mt,
    )
    L = feature_dict["X"].shape[1]
    feature_dict["batch_size"] = 1
    feature_dict["temperature"] = args.temperature
    feature_dict["bias"] = torch.zeros(1, L, 21)
    native_seq = seq_to_string(feature_dict["S"][0].cpu().numpy())
    print(f"    蛋白长度 {L}，native: {native_seq[:50]}...")
    native_charge = net_charge(native_seq, args.pH)  # 提前算（自动补全 + 步骤 5 复用）

    # 2.5 pH-only 自动补全（v3 D1/A9）：未显式传 target → 默认 target=native_charge@pH，
    #     语义"设计一条在该 pH 下保持 native 电荷行为的序列"，完全落在训练分布内；
    #     --no_auto_target_charge 关闭 → 回到旧 flag=0（不指定电荷）对照。
    auto_target = False
    if args.target_charge is None and not args.no_auto_target_charge:
        args.target_charge = native_charge
        auto_target = True
        print(f"    [pH-only 自动补全] target = native_charge@{args.pH} = {native_charge:+.2f}")
    elif args.target_charge is None:
        print("    [pH-only 手动关闭] --no_auto_target_charge：条件 flag=0，不指定电荷（A9 对照）")

    # 3. 模式分支：条件注入（Phase 3） vs 引导采样（Phase 1）
    mode = "phase1_guided"
    cond_encoder = None
    if args.cond_encoder:
        mode = "phase3_conditioned" if args.cond_mode == "conditioned" else "phase3_baseline"
        cond_encoder = load_condition_encoder(args.cond_encoder, device)

        # 电荷校准（默认开）：target_eff = (desired - offset) / gain
        # 抵消条件注入的 ~2.57× 电荷过冲（机制见 condition_defaults.yaml）
        # 自动补全的 native target 不做校准（native 电荷本就在训练分布内，无过冲可补偿）
        target_eff = args.target_charge
        calib_note = "（未指定 target，无校准）"
        if auto_target:
            calib_note = f"（自动补全 target={native_charge:+.2f}，跳过校准）"
        elif args.target_charge is not None and not args.no_calibration:
            with open(_CODE_DIR / "configs" / "condition_defaults.yaml") as f:
                cc = yaml.safe_load(f)["condition_defaults"].get("charge_calibration", {})
            gain, offset = cc.get("gain", 2.57), cc.get("offset", 0.16)
            if cc.get("enabled", True):
                target_eff = (args.target_charge - offset) / gain
                calib_note = (f"target {args.target_charge} → 校准后 target_eff "
                              f"{target_eff:.2f}（gain={gain}, offset={offset}）")
            else:
                calib_note = "（config 中 charge_calibration.enabled=false，未校准）"
        elif args.target_charge is not None:
            calib_note = "（--no_calibration，未校准）"

        cond_vec = make_condition_vector(args.pH, net_charge=target_eff).to(device)
        print(f"[3] 条件注入模式: cond_mode={args.cond_mode}, "
              f"cond_vec={[round(x, 2) for x in cond_vec.tolist()]}")
        print(f"    电荷校准: {calib_note}")
        print(f"[4] 条件注入采样 {args.num_samples} 条候选序列...")
        sequences, charges, pIs = [], [], []
        for i in range(args.num_samples):
            feature_dict["randn"] = torch.randn(1, L)
            # baseline 模式：加载编码器但不注入 → 等价 Phase 1「无引导不感知 pH」对照
            enc_inject = None if args.cond_mode == "baseline" else cond_encoder
            out = conditioned_sample(
                model, enc_inject, feature_dict, cond_vec, device=device,
            )
            seq = seq_to_string(out["S"][0].cpu().numpy())
            sequences.append(seq)
            charges.append(net_charge(seq, args.pH))
            pIs.append(find_pI(seq))
            print(f"    [{i+1:2d}] charge={charges[-1]:+6.2f}  pI={pIs[-1]:5.2f}  {seq[:60]}")
    else:
        # Phase 1 引导采样（结构过滤器 + 动态电荷前瞻 logit bias）
        print(f"[3] 引导设置: pH={args.pH}, target_charge={args.target_charge}, "
              f"preset={args.preset}, strength={args.strength}")
        coords = extract_calpha_coords(protein_dict)
        structure_filter = StructureAwareFilter(coords, config=load_preset(args.preset))
        bias_callback = make_dynamic_callback(
            pH=args.pH, target_charge=args.target_charge,
            structure_filter=structure_filter, strength=args.strength,
        )
        print(f"[4] 引导采样 {args.num_samples} 条候选序列...")
        sampler = GuidedSampler(model, device=device)
        sequences, charges, pIs = [], [], []
        for i in range(args.num_samples):
            feature_dict["randn"] = torch.randn(1, L)
            out = sampler.sample(feature_dict, bias_callback=bias_callback)
            seq = seq_to_string(out["S"][0].cpu().numpy())
            sequences.append(seq)
            charges.append(net_charge(seq, args.pH))
            pIs.append(find_pI(seq))
            print(f"    [{i+1:2d}] charge={charges[-1]:+6.2f}  pI={pIs[-1]:5.2f}  {seq[:60]}")

    # 5. native 对照（native_charge 已在 2.5 提前算好，供自动补全复用）
    native_pI = find_pI(native_seq)
    print(f"[5] native   : charge={native_charge:+6.2f}  pI={native_pI:5.2f}  {native_seq[:60]}")

    # 6. 统计 + 输出
    mean_charge = float(np.mean(charges))
    std_charge = float(np.std(charges))
    print(f"    平均净电荷 = {mean_charge:+.2f} ± {std_charge:.2f}  "
          f"(目标 {args.target_charge})")

    out_dir = Path(args.out_dir) if args.out_dir else (
        _CODE_DIR / "output" / f"guided_{Path(args.pdb).stem}_pH{args.pH}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = out_dir / "seqs.fa"
    with open(fasta_path, "w", encoding="utf-8") as f:
        for i, seq in enumerate(sequences):
            f.write(f">sample_{i+1} pH={args.pH} charge={charges[i]:+.2f} pI={pIs[i]:.2f}\n")
            f.write(seq + "\n")
        f.write(f">native charge={native_charge:+.2f} pI={native_pI:.2f}\n")
        f.write(native_seq + "\n")
    summary = {
        "pdb": args.pdb, "pH": args.pH, "target_charge": args.target_charge,
        "auto_target": auto_target,  # True = target 由 pH-only 自动补全（D1/A9）
        "mode": mode,
        "cond_encoder": str(args.cond_encoder) if args.cond_encoder else None,
        "calibrated": bool(args.cond_encoder and not args.no_calibration),
        "preset": args.preset, "temperature": args.temperature,
        "strength": args.strength, "seed": args.seed, "num_samples": args.num_samples,
        "native_charge": native_charge, "native_pI": native_pI,
        "mean_charge": mean_charge, "std_charge": std_charge,
        "sequences": [
            {"seq": s, "charge": c, "pI": p} for s, c, p in zip(sequences, charges, pIs)
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[6] 输出已保存: {fasta_path}")
    print("完成 ✅")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# v10 自动训练管线：MoMPNN → LigandMPNN 顺序训练 + 双编码器泛化验证。
#
# 背景：v10 三组件（A 条件解耦 + B 表面电荷监督 + C 结构惩罚）治"删减捷径"，
# 双 backbone（MoMPNN 无配体 / LigandMPNN 配体）都要训。GPU 仅 GPU3 空闲 → 顺序。
#
# 用法（后台运行）：
#   bash code/tests/run_v10_pipeline.sh  > log/v10_pipeline.stdout 2>&1 &
#   （内部会等待 MoMPNN 训练完成后自动启动 LigandMPNN，再自动验证）
#
# 产物：
#   output/finetune_v10_mompnn/  +  output/finetune_v10_ligand/   两个编码器
#   output/generalization_v10_mompnn/  +  output/generalization_v10_ligand/  验证
#   output/generalization_v10_*_stats.json                          汇总统计
#   log/v10_pipeline.stdout                                         本脚本日志
set -u

ROOT="/data/nfs/IC/baokun_yu/ConfuMPNN"
CONDA_DIR="${HOME}/miniconda3/etc/profile.d/conda.sh"
PYBIN="/home/baokun_yu/miniconda3/envs/confumpnn/bin/python"
cd "$ROOT"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# ================= 阶段 1：等待 MoMPNN 训练完成 =================
# ⚠️ 教训（2026-08-27）：不能用「进程消失」判断完成——MoMPNN 曾因 dangling symlink
# 崩溃（FileExistsError）导致进程消失被误判"完成"，管线重复启动了 LigandMPNN。
# 正确判据 = **产物 checkpoint 存在**。若进程消失但无 checkpoint → 视为崩溃，稍后重跑。
log "阶段1: 等待 MoMPNN 训练完成（checkpoint 存在则视为完成）..."
MOMPNN_CKPT="output/finetune_v10_mompnn/finetune_epoch030.pt"
while [ ! -f "$MOMPNN_CKPT" ] && pgrep -f "train_finetune.py.*finetune_v10_mompnn" >/dev/null 2>&1; do
    sleep 120
done
if [ -f "$MOMPNN_CKPT" ]; then
    log "阶段1完成: MoMPNN checkpoint 存在"
else
    log "阶段1注意: MoMPNN 进程消失但无 checkpoint（可能崩溃，阶段3.5 将重跑）"
fi

# ================= 阶段 2：启动 LigandMPNN 训练 =================
# ⚠️ 若 LigandMPNN 已在跑（可能由旧管线启动），不重复启动，直接等 checkpoint。
LIG_CKPT="output/finetune_v10_ligand/finetune_epoch030.pt"
if [ -f "$LIG_CKPT" ] || pgrep -f "train_finetune.py.*finetune_v10_ligand" >/dev/null 2>&1; then
    log "阶段2: LigandMPNN 已在运行/已完成，跳过启动"
else
    log "阶段2: 启动 LigandMPNN 训练（GPU3，30 epoch，v10 三组件）..."
    nohup setsid "$PYBIN" code/train_finetune.py \
        --device cuda:3 --epochs 30 --ligand \
        --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
        --labels data/ligand_train/labels.npz --dompdb data/ligand_train/all_pdb \
        --lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 \
        --charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15 \
        --decouple_perturb --decouple_range 12.0 \
        --add_supervision --lambda_add 0.3 --sasa_threshold 0.25 \
        --ph_aware_filter --structure_boost 1.5 \
        --out_dir output/finetune_v10_ligand \
        --log_file log/v10_train_ligand.log --log_progress log/v10_train_ligand_prog.json \
        > log/v10_train_ligand.stdout 2>&1 &
    log "阶段2: LigandMPNN 已后台启动"
fi

# ================= 阶段 3：等待 LigandMPNN 完成（checkpoint 判据）=================
log "阶段3: 等待 LigandMPNN 训练完成..."
while [ ! -f "$LIG_CKPT" ] && pgrep -f "train_finetune.py.*finetune_v10_ligand" >/dev/null 2>&1; do
    sleep 120
done
if [ -f "$LIG_CKPT" ]; then
    log "阶段3完成: LigandMPNN checkpoint 存在"
else
    log "阶段3注意: LigandMPNN 进程消失但无 checkpoint（崩溃），阶段3.5 将重跑"
fi

# ================= 阶段 3.5：检查/重跑 MoMPNN（若因 dangling symlink 等中途崩溃）=================
# MoMPNN 训练曾因 dompdb_pdb 的 dangling symlink 崩溃（FileExistsError，2026-08-27 修复）。
# 若 checkpoint 不存在 → 重跑 MoMPNN（GPU3 现已空闲，因为 LigandMPNN 已完成）。
log "阶段3.5: 检查 MoMPNN 编码器是否存在..."
MOMPNN_CKPT="output/finetune_v10_mompnn/finetune_epoch030.pt"
if [ ! -f "$MOMPNN_CKPT" ]; then
    log "阶段3.5: MoMPNN checkpoint 缺失（可能上次崩溃），重跑训练..."
    "$PYBIN" code/train_finetune.py \
        --device cuda:3 --epochs 30 \
        --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt \
        --labels data/cath/labels_balanced_v7.npz --dompdb data/cath/S40/dompdb \
        --curriculum --perturb_scale 2.0 --curriculum_scale_max 8.0 \
        --lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 \
        --charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15 \
        --decouple_perturb --decouple_range 12.0 \
        --add_supervision --lambda_add 0.3 --sasa_threshold 0.25 \
        --ph_aware_filter --structure_boost 1.5 \
        --out_dir output/finetune_v10_mompnn \
        --log_file log/v10_train_mompnn.log --log_progress log/v10_train_mompnn_prog.json \
        >> log/v10_pipeline.stdout 2>&1
    log "阶段3.5: MoMPNN 重跑完成"
else
    log "阶段3.5: MoMPNN checkpoint 存在，跳过重跑"
fi

# ================= 阶段 3.6：LigandMPNN checkpoint 有效性检查 =================
# ⚠️ 教训（2026-08-27）：LigandMPNN 训练曾因配体数据特定域产生 NaN（epoch 1 起
# total=nan，权重全 NaN）。若 checkpoint 存在但含 NaN，必须**不进入验证**（否则产出
# 全 NaN 的泛化结果），且标记需要重跑。本检查：文件存在 + 权重无 NaN 才放行。
log "阶段3.6: 检查 LigandMPNN checkpoint 有效性..."
LIG_OK="no"
if [ -f "$LIG_CKPT" ]; then
    # 用 python 检查 checkpoint 权重是否含 NaN（0 = 有效）
    N_NAN=$("$PYBIN" -c "
import torch, sys
ck = torch.load('$LIG_CKPT', map_location='cpu')
st = ck.get('condition_encoder_state', ck)
print(sum(int(torch.isnan(v).sum().item()) for v in st.values() if torch.is_tensor(v)))
" 2>/dev/null)
    if [ "$N_NAN" = "0" ]; then
        LIG_OK="yes"
        log "阶段3.6: LigandMPNN checkpoint 有效（无 NaN）"
    else
        log "阶段3.6: ⚠️ LigandMPNN checkpoint 含 NaN（$N_NAN 个）！跳过验证，需重跑训练"
    fi
else
    log "阶段3.6: LigandMPNN checkpoint 不存在，跳过其验证"
fi

# ================= 阶段 4：双编码器泛化验证 =================
log "阶段4: 泛化验证（MoMPNN 编码器，protein 模式）..."
source "$CONDA_DIR" 2>/dev/null
conda activate confumpnn
PYTHONPATH=code timeout 7200 "$PYBIN" code/tests/ligand_v9/validate_generalization.py \
    --manifest data/validation_pdbs/validation_manifest.json \
    --out_dir output/generalization_v10_mompnn \
    --mode protein \
    --backbone auto \
    --cond_encoder output/finetune_v10_mompnn/finetune_epoch030.pt \
    --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt \
    --n 30 --device cuda:3 --pH 7.4
log "阶段4: MoMPNN 验证完成，汇总统计..."
PYTHONPATH=code "$PYBIN" code/tests/ligand_v9/generalization_stats.py \
    --root output/generalization_v10_mompnn \
    --manifest data/validation_pdbs/validation_manifest.json \
    --out output/generalization_v10_mompnn_stats.json

if [ "$LIG_OK" = "yes" ]; then
    log "阶段4: LigandMPNN 编码器验证（both 模式）..."
    PYTHONPATH=code timeout 7200 "$PYBIN" code/tests/ligand_v9/validate_generalization.py \
        --manifest data/validation_pdbs/validation_manifest.json \
        --out_dir output/generalization_v10_ligand \
        --mode both \
        --backbone auto \
        --cond_encoder output/finetune_v10_ligand/finetune_epoch030.pt \
        --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
        --n 30 --device cuda:3 --pH 7.4
    log "阶段4: LigandMPNN 验证完成，汇总统计..."
    PYTHONPATH=code "$PYBIN" code/tests/ligand_v9/generalization_stats.py \
        --root output/generalization_v10_ligand \
        --manifest data/validation_pdbs/validation_manifest.json \
        --out output/generalization_v10_ligand_stats.json
else
    log "阶段4: ⚠️ LigandMPNN checkpoint 无效（NaN/缺失），跳过验证（待重跑训练后手动验证）"
fi

# ================= 阶段 5：完成标记 =================
touch log/v10_pipeline.DONE
log "== v10 全管线完成: 训练 + 双编码器泛化验证 + 统计全部就绪 =="

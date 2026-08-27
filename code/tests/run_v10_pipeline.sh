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
log "阶段1: 等待 MoMPNN 训练完成（PID 存在则轮询）..."
while pgrep -f "train_finetune.py.*finetune_v10_mompnn" >/dev/null 2>&1; do
    sleep 120
done
log "阶段1完成: MoMPNN 训练进程已结束"

# ================= 阶段 2：启动 LigandMPNN 训练 =================
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

# ================= 阶段 3：等待 LigandMPNN 完成 =================
log "阶段3: 等待 LigandMPNN 训练完成..."
while pgrep -f "train_finetune.py.*finetune_v10_ligand" >/dev/null 2>&1; do
    sleep 120
done
log "阶段3完成: LigandMPNN 训练进程已结束"

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

# ================= 阶段 5：完成标记 =================
touch log/v10_pipeline.DONE
log "== v10 全管线完成: 训练 + 双编码器泛化验证 + 统计全部就绪 =="

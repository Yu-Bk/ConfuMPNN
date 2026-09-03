#!/usr/bin/env bash
# v12.2 蛋白模式 4 长蛋白（1A65/1BJ4/13BB/1CDG）两口径 H2 补测（paper gap1 v12.2 基线）
#
# 背景：论文叙事 "v12.2 在长蛋白/深负蛋白上不行 → v12.3 补长蛋白数据后一定程度缓解" 需 v12.2 侧同口径基线。
#   对照 v12.3 agent 已跑产物：output/generalization_v12_3_calib_small/（小样本）、generalization_v12_3_bigglobal/（big-global）。
# 口径：
#   A) 小样本现场标定：build_calibration_small.py（4 蛋白 × 5target × n_per10 = 50 条/蛋白）拟合自身 slope
#      → per-protein 表；validate_generalization.py --calibrate auto 用该表 resample 5 臂 × n30 → H2。
#   B) big-global（未标定）：用只读 output/charge_calibration_v12_2_big.json --calibrate global → resample 5 臂 × n30 → H2。
# 产物根：output/paper_gap1_v122_long/（新目录，绝不复写原校准表 charge_calibration_v12_2*.json）。
# 用法：nohup bash code/tests/ligand_v9/run_v12_2_long_2caliber.sh > log/v12_2_long_2caliber.stdout 2>&1 &
set -u
export PATH="/home/baokun_yu/miniconda3/envs/confumpnn/bin:$PATH"
ROOT="/data/nfs/IC/baokun_yu/ConfuMPNN"
cd "$ROOT"
PY="/home/baokun_yu/miniconda3/envs/confumpnn/bin/python"

MANIFEST="data/validation_pdbs/validation_manifest_v12_2_long.json"
ENC="output/finetune_v12_2/finetune_epoch030.pt"
W="MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt"
BIGCAL="output/charge_calibration_v12_2_big.json"
OUTROOT="output/paper_gap1_v122_long"
SMALL_CAL="$OUTROOT/charge_calibration_v12_2_long_small.json"
SMALL_ROOT="$OUTROOT/generalization_small"
GLOBAL_ROOT="$OUTROOT/generalization_bigglobal"
DEV="${DEV:-cuda:3}"
NPROT=4

log() { echo "[$(date '+%m-%d %H:%M:%S')] $1"; }
count() { find "$1" -name validation.json 2>/dev/null | wc -l; }

mkdir -p "$OUTROOT"

# ===== A 步：小样本现场标定（4 蛋白 × 50 条 = 200 条）=====
if [ ! -s "$SMALL_CAL" ]; then
  log "A 小样本标定启动（4 蛋白 × 5 target × n_per10，$DEV）..."
  PYTHONPATH=code "$PY" code/tests/build_calibration_small.py \
    --manifest "$MANIFEST" --enc "$ENC" --weights "$W" \
    --big_cal "$BIGCAL" --out "$SMALL_CAL" \
    --n_per 10 --device "$DEV" \
    > log/v12_2_long_calib_small.log 2>&1
  log "A 小样本标定完成：$SMALL_CAL"
else
  log "A 小样本标定已存在，跳过：$SMALL_CAL"
fi

# ===== 口径 A：小样本 per-protein 表 resample（4 蛋白 × 5 臂 × n30）=====
log "== 口径 A：小样本现场标定 resample =="
if [ "$(count "$SMALL_ROOT/protein")" -lt "$NPROT" ]; then
  log "小样本 resample 启动（$DEV）..."
  PYTHONPATH=code "$PY" code/tests/ligand_v9/validate_generalization.py \
    --manifest "$MANIFEST" --out_dir "$SMALL_ROOT" --mode protein --backbone auto \
    --cond_encoder "$ENC" --weights "$W" \
    --protein_arms native,n2,p2,n8,p8 \
    --n 30 --device "$DEV" --pH 7.4 \
    --calibrate auto --calibration_file "$SMALL_CAL" \
    > log/v12_2_long_small_sample.log 2>&1
  log "小样本 resample 完成：$(count "$SMALL_ROOT/protein")/$NPROT"
else
  log "小样本 resample 已完成，跳过（$(count "$SMALL_ROOT/protein")/$NPROT）"
fi

# ===== 口径 B：big-global（只读表）resample =====
log "== 口径 B：big-global resample =="
if [ "$(count "$GLOBAL_ROOT/protein")" -lt "$NPROT" ]; then
  log "big-global resample 启动（$DEV）..."
  PYTHONPATH=code "$PY" code/tests/ligand_v9/validate_generalization.py \
    --manifest "$MANIFEST" --out_dir "$GLOBAL_ROOT" --mode protein --backbone auto \
    --cond_encoder "$ENC" --weights "$W" \
    --protein_arms native,n2,p2,n8,p8 \
    --n 30 --device "$DEV" --pH 7.4 \
    --calibrate global --calibration_file "$BIGCAL" \
    > log/v12_2_long_bigglobal_sample.log 2>&1
  log "big-global resample 完成：$(count "$GLOBAL_ROOT/protein")/$NPROT"
else
  log "big-global resample 已完成，跳过（$(count "$GLOBAL_ROOT/protein")/$NPROT）"
fi

touch log/v12_2_long_2caliber.DONE
log "== v12.2 4 长蛋白两口径补测全部完成 =="

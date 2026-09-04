#!/usr/bin/env bash
# Protein-family ablation driver (v12.2 recipe, MoMPNN) — GPU6, 10 epochs each, serial.
# Each run differs from FULL by exactly one module OFF.
set -u
CONDA_PY=/home/baokun_yu/miniconda3/envs/confumpnn/bin/python
ROOT=/data/nfs/IC/baokun_yu/ConfuMPNN
cd "$ROOT"

DEV=cuda:6
EP=10
COMMON="--device $DEV --epochs $EP \
  --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt \
  --labels ablation/data/labels_ablate_prot.npz --dompdb data/cath/S40/dompdb \
  --lambda_c 0.5 --lambda_kl 0.05 \
  --charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15 \
  --decouple_perturb --decouple_range 12.0 \
  --pocket_mode keep --pocket_cutoff 8.0 --pocket_floor 0.7 --pocket_ceil 1.3 --lambda_pocket 0.0"

run_one () {
  local TAG=$1; shift
  local OD="$ROOT/ablation/runs/prot/$TAG"
  mkdir -p "$OD"
  echo "[$(date '+%F %T')] START prot/$TAG"
  # shellcheck disable=SC2086
  PYTHONPATH=code nohup "$CONDA_PY" code/train_finetune.py $COMMON "$@" \
      --out_dir "$OD" --log_file "$OD/train.log" > "$OD/train.stdout" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] DONE prot/$TAG rc=$rc"
  return $rc
}

# FULL: all modules ON
run_one run_FULL \
  --lambda_keep 0.5 \
  --ph_aware_filter --structure_boost 1.5 \
  --v12_supervision --frac_floor 0.5 --gravy_margin 0.4 --lambda_v12 0.2 --lambda_target 0.2 --sasa_threshold 0.25

# -v12 composition supervision (drop --v12_supervision entirely)
run_one run_nov12comp \
  --lambda_keep 0.5 \
  --ph_aware_filter --structure_boost 1.5

# -lambda_target (omit -> default 0.0; keep v12 comp+gravy + everything else)
run_one run_notarget \
  --lambda_keep 0.5 \
  --ph_aware_filter --structure_boost 1.5 \
  --v12_supervision --frac_floor 0.5 --gravy_margin 0.4 --lambda_v12 0.2 --sasa_threshold 0.25

# -ph_aware_filter (drop --ph_aware_filter / --structure_boost)
run_one run_noph \
  --lambda_keep 0.5 \
  --v12_supervision --frac_floor 0.5 --gravy_margin 0.4 --lambda_v12 0.2 --lambda_target 0.2 --sasa_threshold 0.25

# -seq_keep (lambda_keep 0.0)
run_one run_nokeep \
  --lambda_keep 0.0 \
  --ph_aware_filter --structure_boost 1.5 \
  --v12_supervision --frac_floor 0.5 --gravy_margin 0.4 --lambda_v12 0.2 --lambda_target 0.2 --sasa_threshold 0.25

echo "[$(date '+%F %T')] ALL PROT RUNS COMPLETE"

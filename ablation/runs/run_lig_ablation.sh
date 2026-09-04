#!/usr/bin/env bash
# Ligand-family ablation driver (v14 recipe, LigandMPNN atom25) — GPU6, 16 epochs each, serial.
# Each run differs from FULL by exactly one module OFF.
set -u
CONDA_PY=/home/baokun_yu/miniconda3/envs/confumpnn/bin/python
ROOT=/data/nfs/IC/baokun_yu/ConfuMPNN
cd "$ROOT"

DEV=cuda:6
EP=16
COMMON="--device $DEV --epochs $EP --ligand \
  --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
  --labels ablation/data/labels_ablate_lig.npz --dompdb data/ligand_train/all_pdb \
  --lambda_c 0.5 --lambda_kl 0.05 \
  --charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15 \
  --decouple_absolute --decouple_abs_lo=-35.0 --decouple_abs_hi=20.0 \
  --pocket_mode global --pocket_cutoff 8.0 --pocket_floor 0.8 --pocket_ceil 1.3"

run_one () {
  local TAG=$1; shift
  local OD="$ROOT/ablation/runs/lig/$TAG"
  mkdir -p "$OD"
  echo "[$(date '+%F %T')] START lig/$TAG"
  # shellcheck disable=SC2086
  PYTHONPATH=code nohup "$CONDA_PY" code/train_finetune.py $COMMON "$@" \
      --out_dir "$OD" --log_file "$OD/train.log" > "$OD/train.stdout" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] DONE lig/$TAG rc=$rc"
  return $rc
}

# FULL: all modules ON
run_one run_FULL \
  --lambda_keep 0.5 \
  --ph_aware_filter --structure_boost 1.5 \
  --v12_supervision --frac_floor 0.5 --gravy_margin 0.4 --lambda_v12 0.2 --lambda_target 0.2 --sasa_threshold 0.25 \
  --lambda_pocket 0.3

# -v12 composition supervision (drop --v12_supervision)
run_one run_nov12comp \
  --lambda_keep 0.5 \
  --ph_aware_filter --structure_boost 1.5 \
  --lambda_pocket 0.3

# -lambda_target (omit -> default 0; keep v12 comp+gravy, A1)
run_one run_notarget \
  --lambda_keep 0.5 \
  --ph_aware_filter --structure_boost 1.5 \
  --v12_supervision --frac_floor 0.5 --gravy_margin 0.4 --lambda_v12 0.2 --sasa_threshold 0.25 \
  --lambda_pocket 0.3

# -A1 pocket count (lambda_pocket 0; keep global partitioning semantics for lambda_target)
run_one run_noA1 \
  --lambda_keep 0.5 \
  --ph_aware_filter --structure_boost 1.5 \
  --v12_supervision --frac_floor 0.5 --gravy_margin 0.4 --lambda_v12 0.2 --lambda_target 0.2 --sasa_threshold 0.25 \
  --lambda_pocket 0.0

# -ph_aware_filter
run_one run_noph \
  --lambda_keep 0.5 \
  --v12_supervision --frac_floor 0.5 --gravy_margin 0.4 --lambda_v12 0.2 --lambda_target 0.2 --sasa_threshold 0.25 \
  --lambda_pocket 0.3

# -seq_keep (lambda_keep 0.0)
run_one run_nokeep \
  --lambda_keep 0.0 \
  --ph_aware_filter --structure_boost 1.5 \
  --v12_supervision --frac_floor 0.5 --gravy_margin 0.4 --lambda_v12 0.2 --lambda_target 0.2 --sasa_threshold 0.25 \
  --lambda_pocket 0.3

echo "[$(date '+%F %T')] ALL LIG RUNS COMPLETE"

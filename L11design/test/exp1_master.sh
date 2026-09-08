#!/bin/bash
# Exp1 master orchestrator (corrected design 2026-09-09).
#   waits for a genuinely idle GPU -> sample 4 groups (n100) + raw diag (n30)
#   -> postprocess -> ESMFold+TM (GPU) -> Tm+Sol (CPU) -> analyze.
# Resume-safe: each step skips if its output exists.
set -u
ROOT=/data/nfs/IC/baokun_yu/ConfuMPNN
LOG=$ROOT/L11design/test/log
mkdir -p "$LOG"

echo "[master] start $(date)"
# ---- wait for idle GPU (util<30% AND free mem>15GB) ----
while :; do
  gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader \
        | awk -F', ' '{gsub("%","",$2); gsub(" MiB","",$3); gsub(" MiB","",$4);
                       if ($2<30 && ($4-$3)>15000) {print $1; exit}}')
  if [ -n "$gpu" ]; then
    echo "[master] picked GPU $gpu at $(date)"
    break
  fi
  sleep 60
done
export CUDA_VISIBLE_DEVICES=$gpu

# ---- 1) sampling ----
bash $ROOT/L11design/test/exp1_run_sampling.sh all || { echo "[master] sampling failed"; exit 1; }

# ---- 2) postprocess ----
$HOME/miniconda3/envs/confumpnn/bin/python $ROOT/L11design/test/exp1_postprocess.py all \
  || { echo "[master] postprocess failed"; exit 1; }

# ---- 3) ESMFold + TM on same GPU ----
bash $ROOT/L11design/test/exp1_run_fold.sh all || echo "[master] fold/tm had errors"

# ---- 4) CPU scores: TemBERTure + Protein-Sol ----
bash $ROOT/L11design/test/exp1_run_cpu_scores.sh all || echo "[master] cpu scores had errors"

# ---- 5) analyze ----
$HOME/miniconda3/envs/confumpnn/bin/python $ROOT/L11design/test/exp1_analyze.py || echo "[master] analyze had errors"

echo "[master] ALL DONE $(date)"

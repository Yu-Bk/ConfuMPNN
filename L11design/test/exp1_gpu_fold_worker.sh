#!/bin/bash
# Exp1 GPU worker: waits for an idle GPU AND for each group's cleaned fasta (postprocess),
# then runs ESMFold + TM-score for that group. Tm/Sol (CPU) are kicked off when data is ready.
# Resume-safe. Launch as a background job.
set -u
ROOT=/data/nfs/IC/baokun_yu/ConfuMPNN
DATAROOT=$ROOT/L11design/data/exp1
LOG=$ROOT/L11design/test/log
mkdir -p "$LOG"
GRP_LIST="prot_C1 prot_C2 lig_C1 lig_C2"

# ---- wait for idle GPU ----
while :; do
  gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader \
        | awk -F', ' '{gsub("%","",$2); gsub(" MiB","",$3); gsub(" MiB","",$4);
                       if ($2<30 && ($4-$3)>15000) {print $1; exit}}')
  if [ -n "$gpu" ]; then
    echo "[gpu-worker] picked GPU $gpu at $(date)"; break
  fi
  sleep 60
done
export CUDA_VISIBLE_DEVICES=$gpu

# ---- fold groups as they become ready ----
for grp in $GRP_LIST; do
  echo "[gpu-worker] waiting data for $grp at $(date)"
  while [ ! -f "$DATAROOT/$grp/seqs.fa" ]; do sleep 60; done
  bash $ROOT/L11design/test/exp1_run_fold.sh "$grp" || echo "[gpu-worker] fold $grp had errors"
  # TM/Tm/Sol CPU scores for this group (independent of GPU; run here to pipeline)
  bash $ROOT/L11design/test/exp1_run_cpu_scores.sh "$grp" || echo "[gpu-worker] cpu scores $grp had errors"
done

echo "[gpu-worker] FOLD+CPU SCORES DONE $(date)"

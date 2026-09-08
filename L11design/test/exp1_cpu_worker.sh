#!/bin/bash
# Exp1 CPU worker: samples all groups on CPU (thread-capped) then postprocesses.
# GPU work (ESMFold) is handled by exp1_gpu_fold_worker.sh once a card frees.
# Resume-safe: skips groups whose seqs.fa already exists.
set -u
ROOT=/data/nfs/IC/baokun_yu/ConfuMPNN
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-48}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-48}
export CUDA_VISIBLE_DEVICES=""
echo "[cpu-worker] start $(date) threads=$OMP_NUM_THREADS"
bash $ROOT/L11design/test/exp1_run_sampling.sh all || { echo "[cpu-worker] sampling failed"; exit 1; }
$HOME/miniconda3/envs/confumpnn/bin/python $ROOT/L11design/test/exp1_postprocess.py all \
  || { echo "[cpu-worker] postprocess failed"; exit 1; }
echo "[cpu-worker] DONE $(date)"

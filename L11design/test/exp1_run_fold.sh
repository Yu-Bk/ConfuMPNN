#!/bin/bash
# Exp1 ESMFold + TM-score (needs GPU, run in confumpnn-esmfold env via full python path).
# Usage: bash L11design/test/exp1_run_fold.sh [grp|all] [--skip-fold]
# ENV: CUDA_VISIBLE_DEVICES selects a free GPU.
set -u
ROOT=/data/nfs/IC/baokun_yu/ConfuMPNN
ESMPY=$HOME/miniconda3/envs/confumpnn-esmfold/bin/python
CONFPY=$HOME/miniconda3/envs/confumpnn/bin/python
DATAROOT=$ROOT/L11design/data/exp1
REF=$ROOT/L11design/input/L11.pdb
LOG=$ROOT/L11design/test/log
mkdir -p "$LOG"
requested="${1:-all}"

for grp in prot_C1 prot_C2 lig_C1 lig_C2; do
  if [ "$requested" != "all" ] && [ "$requested" != "$grp" ]; then continue; fi
  d=$DATAROOT/$grp
  fa=$d/seqs.fa
  [ -f "$fa" ] || { echo "[missing fasta] $grp"; continue; }
  # 1) ESMFold
  if [ -f "$d/plddt.csv" ] && [ -d "$d/folds" ] && [ -n "$(ls -A $d/folds 2>/dev/null)" ]; then
    echo "[skip-fold] $grp"
  else
    echo "== ESMFold $grp =="
    timeout 7200 $ESMPY $ROOT/code/tests/esmfold_score.py --fasta "$fa" \
        --out "$d/plddt.csv" --outdir "$d/folds" --device cuda > "$LOG/esmfold_$grp.log" 2>&1
    echo "  rc=$? (tail:)"
    tail -3 "$LOG/esmfold_$grp.log"
  fi
  # 2) TM-score vs native chain I
  if [ -f "$d/tm.csv" ]; then
    echo "[skip-tm] $grp"
  else
    echo "== TMscore $grp =="
    timeout 3600 $CONFPY $ROOT/code/tests/tm_score.py --folds "$d/folds" --ref "$REF" \
        --out "$d/tm.csv" > "$LOG/tm_$grp.log" 2>&1
    echo "  rc=$? (tail:)"
    tail -3 "$LOG/tm_$grp.log"
  fi
done
echo "FOLD+TM DONE"

#!/bin/bash
# L11 Exp2 TM-score/RMSD（US-align，CPU）：data/exp2/*/folds vs L11_chainI.pdb
# 前置：ESMFold 已把回折 PDB 写入 data/exp2/*/folds/
# 用法：bash L11design/test/exp2_tm_run.sh
set -u
source /home/baokun_yu/miniconda3/etc/profile.d/conda.sh
cd /data/nfs/IC/baokun_yu/ConfuMPNN
mkdir -p log
REF=L11design/input/L11_chainI.pdb
PYC=/home/baokun_yu/miniconda3/envs/confumpnn/bin/python

for d in L11design/data/exp2/*/; do
  folds="$d/folds"
  [ -d "$folds" ] || { echo "  skip $d（无 folds）"; continue; }
  out="$d/tm.csv"
  [ -f "$out" ] && [ -s "$out" ] && { echo "  skip $d（已有 tm.csv）"; continue; }
  echo "  TM: $d"
  $PYC code/tests/tm_score.py --folds "$folds" --ref "$REF" --out "$out" >> log/exp2_tm.log 2>&1
done
echo "TM SCORES DONE"

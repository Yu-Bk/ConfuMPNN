#!/bin/bash
# Phase 3 防失控判据：条件注入生成序列的四指标打分（pLDDT/TM/%sol/Tm）
# 前置：phase3 序列已生成（{TARGET_DIR}/{pdb}/seqs.fa）
# 用法：bash code/tests/phase3_antidrift_score.sh [TARGET_DIR]
#       # 默认 code/output/phase3_antidrift；校准版用 code/output/phase3_antidrift_cal
cd /data/nfs/IC/baokun_yu/ConfuMPNN
mkdir -p code/log

PY_CONF=/home/baokun_yu/miniconda3/envs/confumpnn/bin/python
PY_ESM=/home/baokun_yu/miniconda3/envs/confumpnn-esmfold/bin/python
PY_TEMP=/home/baokun_yu/miniconda3/envs/confumpnn-temberture/bin/python
TARGET_DIR=${1:-code/output/phase3_antidrift}

declare -A REF
REF[1BC8]=code/input/1BC8_chainC.pdb
REF[1CRN]=code/input/1CRN.pdb
REF[1UBQ]=code/input/1UBQ.pdb
REF[2LZM]=code/input/2LZM.pdb

echo "=== [1/4] ESMFold 回折 + pLDDT（confumpnn-esmfold，GPU）==="
$PY_ESM code/tests/esmfold_score.py --input-dir $TARGET_DIR --device cuda
echo "  ESMFold DONE $(date +%H:%M:%S)"

echo "=== [2/4] TM-score 自洽性（USalign）==="
for pdb in 1BC8 1CRN 1UBQ 2LZM; do
  cond=$TARGET_DIR/$pdb
  if [ -d "$cond/folds" ]; then
    $PY_CONF code/tests/tm_score.py --folds "$cond/folds" --ref "${REF[$pdb]}" \
      --out "$cond/tm.csv" 2>/dev/null
    echo "  $pdb tm DONE"
  fi
done

echo "=== [3/4] Protein-Sol %sol（系统 python3）==="
for fa in $TARGET_DIR/*/seqs.fa; do
  python3 protein_sol_mcp/scripts/protein_sol_predict.py "$fa" 2>/dev/null
  echo "  $(dirname "$fa") sol DONE"
done

echo "=== [4/4] TemBERTure Tm（confumpnn-temberture，每 PDB 一进程并行）==="
PIDS=()
for pdb in 1BC8 1CRN 1UBQ 2LZM; do
  OMP_NUM_THREADS=4 $PY_TEMP code/tests/temberture_score.py \
    --input-dir $TARGET_DIR/$pdb --out $TARGET_DIR/$pdb/seqs.fa.tm.csv &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait $pid; done
echo "  TemBERTure DONE $(date +%H:%M:%S)"

echo "=== ALL PHASE3 SCORING DONE ==="

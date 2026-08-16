#!/bin/bash
# Phase 3 防失控扩样本（n=20）四指标打分管线：pLDDT / TM-score / %sol / Tm
# 结构：{TARGET_DIR}/{pdb}/{A_base,A_cond,B_base,B_cond}/seqs.fa
#   采样由 tests/phase3_antidrift_extend.py 生成（对称配对协议）
# 用法：bash code/tests/phase3_antidrift_n20_score.sh [TARGET_DIR] [GPU_ID]
#       # 默认 code/output/phase3_antidrift_n20，GPU 1
cd /data/nfs/IC/baokun_yu/ConfuMPNN
mkdir -p code/log

PY_CONF=/home/baokun_yu/miniconda3/envs/confumpnn/bin/python
PY_ESM=/home/baokun_yu/miniconda3/envs/confumpnn-esmfold/bin/python
PY_TEMP=/home/baokun_yu/miniconda3/envs/confumpnn-temberture/bin/python
TARGET_DIR=${1:-code/output/phase3_antidrift_n20}
GPU=${2:-1}

declare -A REF
REF[1BC8]=code/input/1BC8_chainC.pdb
REF[1CRN]=code/input/1CRN.pdb
REF[1UBQ]=code/input/1UBQ.pdb
REF[2LZM]=code/input/2LZM.pdb

echo "=== [1/4] ESMFold 回折 + pLDDT（confumpnn-esmfold，递归扫描所有 arm 的 seqs.fa）==="
CUDA_VISIBLE_DEVICES=$GPU $PY_ESM code/tests/esmfold_score.py \
  --input-dir $TARGET_DIR --device cuda
echo "  ESMFold DONE $(date +%H:%M:%S)"

echo "=== [2/4] TM-score 自洽性（USalign，每 arm 的 folds/ vs 原骨架）==="
for armdir in $TARGET_DIR/*/*/; do
  pdb=$(basename $(dirname "$armdir"))
  if [ -d "$armdir/folds" ]; then
    $PY_CONF code/tests/tm_score.py --folds "$armdir/folds" \
      --ref "${REF[$pdb]}" --out "$armdir/tm.csv" 2>/dev/null
    echo "  $(basename $(dirname "$armdir"))/$(basename "$armdir") tm DONE"
  fi
done

echo "=== [3/4] Protein-Sol %sol（系统 python3，递归 arm 目录）==="
for fa in $TARGET_DIR/*/*/seqs.fa; do
  python3 protein_sol_mcp/scripts/protein_sol_predict.py "$fa" 2>/dev/null
  echo "  $(basename $(dirname "$fa")) sol DONE"
done

echo "=== [4/4] TemBERTure Tm（confumpnn-temberture，每 arm 一进程并行）==="
PIDS=()
for armdir in $TARGET_DIR/*/*/; do
  OMP_NUM_THREADS=4 $PY_TEMP code/tests/temberture_score.py \
    --input-dir "$armdir" --out "$armdir/seqs.fa.tm.csv" &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait $pid; done
echo "  TemBERTure DONE $(date +%H:%M:%S)"

echo "=== ALL PHASE3 N20 SCORING DONE ==="

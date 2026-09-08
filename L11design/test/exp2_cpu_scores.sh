#!/bin/bash
# L11 Exp2 CPU 打分（不需 GPU）：
#   1) TemBERTure Tm  （confumpnn-temberture） → data/exp2/*/seqs.fa.tm.csv
#   2) Protein-Sol %sol（confumpnn python 调 wrapper，串行）→ data/exp2/*/seqs.fa-protein_sol.csv
# 用法：bash L11design/test/exp2_cpu_scores.sh
set -u
source /home/baokun_yu/miniconda3/etc/profile.d/conda.sh
cd /data/nfs/IC/baokun_yu/ConfuMPNN
mkdir -p log
PYT=/home/baokun_yu/miniconda3/envs/confumpnn-temberture/bin/python
PYC=/home/baokun_yu/miniconda3/envs/confumpnn/bin/python

echo "=== [1/2] TemBERTure Tm ==="
$PYT code/tests/temberture_score.py --input-dir L11design/data/exp2 > log/exp2_temberture.log 2>&1
echo "  Tm done (log/exp2_temberture.log)"

echo "=== [2/2] Protein-Sol ==="
for fa in L11design/data/exp2/*/seqs.fa; do
  d=$(dirname "$fa")
  out="$d/seqs.fa-protein_sol.csv"
  if [ -f "$out" ] && [ -s "$out" ]; then
    echo "  skip $d（已有 csv）"
    continue
  fi
  echo "  sol: $d"
  $PYC protein_sol_mcp/scripts/protein_sol_predict.py "$fa" >> log/exp2_protsol.log 2>&1
  if [ ! -f "$out" ]; then
    echo "  !! Protein-Sol 未产出 $out（见 log/exp2_protsol.log）"
  fi
done
echo "ALL CPU SCORES DONE"

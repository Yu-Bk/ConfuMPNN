#!/bin/bash
# L11 Exp3/Exp4 CPU 打分（不需 GPU）：
#   1) TemBERTure Tm （confumpnn-temberture） → data/exp3|exp4/*/seqs.fa.tm.csv
#   2) Protein-Sol %sol（confumpnn python 调 wrapper，串行）→ data/exp3|exp4/*/seqs.fa-protein_sol.csv
# 用法：bash L11design/test/exp34_cpu_scores.sh
set -u
source /home/baokun_yu/miniconda3/etc/profile.d/conda.sh
cd /data/nfs/IC/baokun_yu/ConfuMPNN
mkdir -p log
PYT=/home/baokun_yu/miniconda3/envs/confumpnn-temberture/bin/python
PYC=/home/baokun_yu/miniconda3/envs/confumpnn/bin/python

echo "=== [1/2] TemBERTure Tm ==="
$PYT code/tests/temberture_score.py --input-dir L11design/data/exp3 > log/exp34_temberture_exp3.log 2>&1 &
P1=$!
$PYT code/tests/temberture_score.py --input-dir L11design/data/exp4 > log/exp34_temberture_exp4.log 2>&1 &
P2=$!
wait $P1; echo "  exp3 Tm done (log/exp34_temberture_exp3.log)"
wait $P2; echo "  exp4 Tm done (log/exp34_temberture_exp4.log)"

echo "=== [2/2] Protein-Sol ==="
for fa in L11design/data/exp3/*/seqs.fa L11design/data/exp4/*/seqs.fa; do
  [ -f "$fa" ] || continue
  d=$(dirname "$fa")
  out="$d/seqs.fa-protein_sol.csv"
  if [ -f "$out" ] && [ -s "$out" ]; then
    echo "  skip $d（已有 csv）"
    continue
  fi
  echo "  sol: $d"
  $PYC protein_sol_mcp/scripts/protein_sol_predict.py "$fa" >> log/exp34_protsol.log 2>&1
  if [ ! -f "$out" ]; then
    echo "  !! Protein-Sol 未产出 $out（见 log/exp34_protsol.log）"
  fi
done
echo "ALL CPU SCORES DONE"

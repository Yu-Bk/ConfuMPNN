#!/bin/bash
# Exp1 CPU scores: TemBERTure Tm (confumpnn-temberture env) then Protein-Sol (confumpnn env).
# Protein-Sol must run serially (shares a working dir).
# Usage: bash L11design/test/exp1_run_cpu_scores.sh [grp|all] [--tm-only|--sol-only]
set -u
ROOT=/data/nfs/IC/baokun_yu/ConfuMPNN
TMPY=$HOME/miniconda3/envs/confumpnn-temberture/bin/python
CONFPY=$HOME/miniconda3/envs/confumpnn/bin/python
DATAROOT=$ROOT/L11design/data/exp1
SOLPY=$ROOT/protein_sol_mcp/scripts/protein_sol_predict.py
LOG=$ROOT/L11design/test/log
mkdir -p "$LOG"
requested="${1:-all}"
mode="${2:-all}"   # all | --tm-only | --sol-only

do_tm () {
  grp=$1; d=$DATAROOT/$grp; fa=$d/seqs.fa
  [ -f "$fa" ] || { echo "[missing fasta] $grp"; return; }
  if [ -f "$fa.tm.csv" ]; then echo "[skip-tm] $grp"; return; fi
  echo "== TemBERTure $grp =="
  timeout 7200 $TMPY $ROOT/code/tests/temberture_score.py --fasta "$fa" --out "$fa.tm.csv" \
      > "$LOG/tm_$grp.log" 2>&1
  echo "  rc=$? tail:"; tail -2 "$LOG/tm_$grp.log"
}

do_sol () {
  grp=$1; d=$DATAROOT/$grp; fa=$d/seqs.fa
  [ -f "$fa" ] || { echo "[missing fasta] $grp"; return; }
  outtxt="$fa-protein_sol_prediction.txt"
  if [ -f "$outtxt" ]; then echo "[skip-sol] $grp"; return; fi
  echo "== Protein-Sol $grp =="
  (cd "$ROOT" && timeout 10800 $CONFPY "$SOLPY" "$fa" > "$LOG/sol_$grp.log" 2>&1)
  echo "  rc=$? tail:"; tail -2 "$LOG/sol_$grp.log"
}

for grp in prot_C1 prot_C2 lig_C1 lig_C2; do
  if [ "$requested" != "all" ] && [ "$requested" != "$grp" ]; then continue; fi
  if [ "$mode" = "all" ] || [ "$mode" = "--tm-only" ]; then do_tm "$grp"; fi
done
# Sol strictly serial (shared perl working dir)
for grp in prot_C1 prot_C2 lig_C1 lig_C2; do
  if [ "$requested" != "all" ] && [ "$requested" != "$grp" ]; then continue; fi
  if [ "$mode" = "all" ] || [ "$mode" = "--sol-only" ]; then do_sol "$grp"; fi
done
echo "CPU SCORES DONE"

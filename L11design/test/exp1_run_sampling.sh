#!/bin/bash
# Exp1 (corrected 2026-09-09): 2 modes x 2 pH-conditions, n=100 per group + raw diag n=30.
# Usage:
#   bash L11design/test/exp1_run_sampling.sh <grp>   # run one group (or 'all')
# Env: CUDA_VISIBLE_DEVICES must select a free GPU (device chosen automatically by torch).
# Dry-runs validated: protein mode + ligand mode both parse fixed residues correctly.
set -u
ROOT=/data/nfs/IC/baokun_yu/ConfuMPNN
PY=$HOME/miniconda3/envs/confumpnn/bin/python
MAN=$ROOT/L11design/test/exp1_groups.json
OUTROOT=$ROOT/L11design/output/exp1
LOG=$ROOT/L11design/test/log
mkdir -p "$LOG"

requested="${1:-all}"
RUN_N=${RUN_N:-100}
RAW_N=${RAW_N:-30}
FIX="I3 I5 I9 I34 I35 I89 I124 I131 I134 I135"

run_one () {
  grp=$1
  n=$2
  calibflag=$3        # "" = use calibration file; "off" = --calibrate off
  sub=${4:-$grp}      # output subdir (default group)
  cfg=$(python3 -c "import json;print(json.load(open('$MAN'))['groups']['$grp'])" )
  # shellcheck disable=SC2207
  mode=$(python3 -c "import json;print(json.load(open('$MAN'))['groups']['$grp']['mode'])")
  input=$(python3 -c "import json;print(json.load(open('$MAN'))['groups']['$grp']['input'])")
  pH=$(python3 -c "import json;print(json.load(open('$MAN'))['groups']['$grp']['pH'])")
  tgt=$(python3 -c "import json;print(json.load(open('$MAN'))['groups']['$grp']['target'])")
  seed=$(python3 -c "import json;print(json.load(open('$MAN'))['groups']['$grp']['seed'])")
  weights=$(python3 -c "import json;print(json.load(open('$MAN'))['groups']['$grp']['weights'])")
  enc=$(python3 -c "import json;print(json.load(open('$MAN'))['groups']['$grp']['cond_encoder'])")
  calib=$(python3 -c "import json;print(json.load(open('$MAN'))['groups']['$grp']['calib'])")

  outdir=$OUTROOT/$sub
  mkdir -p "$outdir"
  if [ -f "$outdir/seqs.fa" ] && [ -f "$outdir/summary.json" ]; then
    echo "[skip] $sub already has seqs.fa"
    return 0
  fi

  cmd="$PY $ROOT/code/run_guided.py --pdb $ROOT/$input --pH $pH --target_charge $tgt \
       --weights $ROOT/$weights --cond_encoder $ROOT/$enc \
       --fixed_residues \"$FIX\" --num_samples $n --seed $seed \
       --out_dir $outdir"
  if [ "$calibflag" = "off" ]; then
    cmd="$cmd --calibrate off"
  else
    cmd="$cmd --calibration_file $ROOT/$calib"
  fi
  echo "== $sub (mode=$mode pH=$pH target=$tgt n=$n calib='$calibflag') =="
  echo "CMD: $cmd"
  logf="$LOG/sample_${sub//\//_}.log"
  eval "$cmd" > "$logf" 2>&1
  rc=$?
  tail -6 "$LOG/sample_$sub.log"
  if [ $rc -ne 0 ]; then echo "!! $sub FAILED rc=$rc"; return 1; fi
  echo "[done] $sub"
  return 0
}

for grp in prot_C1 prot_C2 lig_C1 lig_C2; do
  if [ "$requested" != "all" ] && [ "$requested" != "$grp" ]; then continue; fi
  run_one "$grp" "$RUN_N" "" "$grp" || exit 1
  run_one "$grp" "$RAW_N" "off" "rawdiag/$grp" || exit 1
done
echo "ALL SAMPLING DONE"

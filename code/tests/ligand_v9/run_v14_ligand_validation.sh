#!/usr/bin/env bash
# v14 配体完整验证链（coverage=in 主判据，1A65 boundary 单列）
#
# 环节：① 配体诊断 slope → ② 校准表（v14 新模型重拟）→ ③ 泛化采样 n50（in 10 蛋白 × 5 臂）
#       → ④ 组成分析（0.7-1.3×）→ ⑤ ESMFold H1 + TM → ⑥ H2 统计
#       → ⑦ PROPKA H4 → ⑧ H3 全臂 → ⑨ Tm/Sol S2。最后单独跑 boundary(1A65)。
#
# 判据（coverage=in）：校准后 slope [0.9,1.15]、H2≥72%、H1 TM≥0.7、组成健康、
#   H3 ≤ 基线+5pp、Tm/Sol S2 不劣于 v12.2 配体(9/50) / v13(17/50)。1A65 单列不进判据。
#
# 用法：nohup bash code/tests/ligand_v9/run_v14_ligand_validation.sh > log/v14_ligand_validation.stdout 2>&1 &
set -u
export PATH="/home/baokun_yu/miniconda3/envs/confumpnn/bin:$PATH"
ROOT="/data/nfs/IC/baokun_yu/ConfuMPNN"
cd "$ROOT"
PY="/home/baokun_yu/miniconda3/envs/confumpnn/bin/python"
PYE="/home/baokun_yu/miniconda3/envs/confumpnn-esmfold/bin/python"
PYT="/home/baokun_yu/miniconda3/envs/confumpnn-temberture/bin/python"
DIAG_GPU="${DIAG_GPU:-cuda:4}"
SAMP_GPU="${SAMP_GPU:-cuda:4}"
ESM_GPU="${ESM_GPU:-cuda:5}"
MAN_IN="data/validation_pdbs/validation_manifest_v14_in.json"
MAN_BD="data/validation_pdbs/validation_manifest_v14_boundary.json"
ENC="output/finetune_ligand_v14_rna/finetune_epoch050.pt"
W="LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt"
OUT="output/generalization_ligand_v14"
TMROOT="output/tm_sol_ligand_v14"
DIAG="output/v14_ligand_diag_response.json"
CAL="output/charge_calibration_v14_ligand.json"
CAL_IN="output/charge_calibration_v14_ligand_in.json"
NATOMS=25

log() { echo "[$(date '+%H:%M:%S')] $1"; }
count() { find "$1" -name "$2" 2>/dev/null | wc -l; }

[ -f "$ENC" ] || { log "❌ checkpoint 缺失 $ENC"; exit 1; }

# ============ ① 配体诊断 slope（in manifest 10 + trainish 8）============
if [ ! -f "$DIAG" ]; then
  log "① 配体诊断启动（$DIAG_GPU）..."
  PYTHONPATH=code timeout 21600 "$PY" index/v10_repair/v10_diag_response_curve.py \
    --backbone ligand_mpnn --cond_encoder "$ENC" --weights "$W" \
    --pdb-list log/v14_ligand_trainish.list --manifest "$MAN_IN" \
    --targets=-34,-30,-25,-20,-15,-10,-5,0,5,10,18 --include_native --n 20 \
    --num_ligand_atoms $NATOMS \
    --device "$DIAG_GPU" --out "$DIAG" > log/v14_ligand_diag.log 2>&1
else
  log "① 诊断已有，跳过"
fi
[ -f "$DIAG" ] || { log "❌ ① 诊断失败，见 log/v14_ligand_diag.log"; exit 1; }

# ============ ② 校准表（in 侧；全 diag 拟合 global，per-protein 记 10 in）============
if [ ! -f "$CAL" ]; then
  log "② 建校准表..."
  PYTHONPATH=code "$PY" index/v10_repair/build_calibration.py \
    --diag "$DIAG" --label v14_ligand --out "$CAL"
fi
[ -f "$CAL" ] || { log "❌ ② 校准失败"; exit 1; }

# ============ ③ 泛化采样 n50（in 10 蛋白 × 5 臂，per-protein 校准）============
if [ "$(count "$OUT/ligand" validation.json)" -lt 10 ]; then
  log "③ 泛化采样 n50（$SAMP_GPU）..."
  PYTHONPATH=code timeout 28800 "$PY" code/tests/ligand_v9/validate_generalization.py \
    --manifest "$MAN_IN" --out_dir "$OUT" --mode ligand --backbone auto \
    --cond_encoder "$ENC" --weights "$W" \
    --n 50 --device "$SAMP_GPU" --pH 7.4 \
    --calibrate auto --calibration_file "$CAL" \
    > log/v14_val_sample.log 2>&1
fi
NVAL=$(count "$OUT/ligand" validation.json)
[ "$NVAL" -ge 10 ] || { log "❌ ③ 泛化采样未完成（$NVAL/10），见 log/v14_val_sample.log"; exit 1; }
log "③ 采样完成：$NVAL/10"

# ============ ④ 组成分析 ============
log "④ 组成分析..."
PYTHONPATH=code "$PY" code/tests/ligand_v9/compare_comp_ligand.py \
  --gen-root "$OUT/ligand" --out output/v14_ligand_comp.json

# ============ ⑤ H1 折叠：ESMFold + TM-score ============
if [ "$(count "$OUT/ligand" plddt.csv)" -lt 50 ]; then
  log "⑤ 回折（$ESM_GPU）..."
  timeout 21600 "$PYE" code/tests/esmfold_score.py --input-dir "$OUT/ligand" --device "$ESM_GPU" \
    > log/v14_val_esmfold.log 2>&1
fi
log "⑤ TM-score..."
for arm_dir in $(find "$OUT/ligand" -type d -name "arm_*" 2>/dev/null); do
  [ -f "$arm_dir/tm.csv" ] && continue
  rel="${arm_dir#"$OUT"/ligand/}"
  pdb="$(echo "$rel" | cut -d/ -f1)"
  [ ! -d "$arm_dir/folds" ] && { log "⚠️ $rel folds 缺失"; continue; }
  timeout 600 "$PY" code/tests/tm_score.py --folds "$arm_dir/folds" \
    --ref "$OUT/ref/${pdb}_ref.pdb" --out "$arm_dir/tm.csv" \
    >> log/v14_val_tm.log 2>&1
done

# ============ ⑥ H2 统计 ============
log "⑥ 统计（H2/GRAVY/RMSD）..."
PYTHONPATH=code "$PY" code/tests/ligand_v9/generalization_stats.py \
  --root "$OUT" --manifest "$MAN_IN" --out output/v14_ligand_gen_stats.json \
  > log/v14_val_stats.log 2>&1

# ============ ⑦ PROPKA H4（in 代表：1BJ4/21KL_A/3MXB_A，native/n8）============
log "⑦ PROPKA 复核..."
PYTHONPATH=code "$PY" - <<'PYEOF'
import json, os, subprocess
OUT = 'output/generalization_ligand_v14/ligand'
PY = '/home/baokun_yu/miniconda3/envs/confumpnn/bin/python'
PROTS = ['1BJ4', '21KL_A', '3MXB_A']
outdir = 'output/propka_v14_ligand'
os.makedirs(outdir, exist_ok=True)
for p in PROTS:
    vj = f'{OUT}/{p}/validation.json'
    if not os.path.exists(vj):
        print('SKIP', p); continue
    q = json.load(open(vj))['native_charge']
    for a, tag in [('native', q), ('n8', q - 8)]:
        folds = f'{OUT}/{p}/pH7.4/arm_{a}/folds'
        if not os.path.isdir(folds):
            print('SKIP folds', p, a); continue
        out = f'{outdir}/{p}_{a}.json'
        if os.path.exists(out):
            continue
        subprocess.run([PY, 'code/tests/propka_charge_check.py', '--pdb', folds,
                        '--pH', '7.4', '--target', str(round(tag)), '--out', out], check=False)
        print('done', out)
PYEOF

# ============ ⑧ H3 全臂（in）============
mkdir -p "$TMROOT/ref_native" "$TMROOT/uncond"
PYTHONPATH=code "$PY" - <<'PYEOF'
import json
from pathlib import Path
ROOT = Path('/data/nfs/IC/baokun_yu/ConfuMPNN')
OUT = ROOT / 'output/generalization_ligand_v14/ligand'
REF = ROOT / 'output/tm_sol_ligand_v14/ref_native'
man = json.load(open(ROOT / 'data/validation_pdbs/validation_manifest_v14_in.json'))
for it in man['items']:
    p = it['pdb']
    fa = OUT / p / 'pH7.4' / 'arm_native' / 'seqs.fa'
    if not fa.exists():
        continue
    native = None
    lines = open(fa).read().splitlines()
    for i, l in enumerate(lines):
        if l.startswith('>native') and i + 1 < len(lines):
            native = lines[i + 1].strip(); break
    if native:
        (REF / f'{p}_native.fa').write_text(f'>{p}_native\n{native}\n')
        print('ref_native:', p)
PYEOF
if [ "$(ls "$TMROOT/uncond" | wc -l)" -lt 10 ]; then
  log "⑧ uncond 采样（$SAMP_GPU）..."
  PYTHONPATH=code timeout 21600 "$PY" code/tests/ligand_v9/sample_unconditioned_ligand.py \
    --manifest "$MAN_IN" --out_dir "$TMROOT/uncond" \
    --cond_encoder "$ENC" --weights "$W" \
    --n 30 --device "$SAMP_GPU" --pH 7.4 > log/v14_uncond_sample.log 2>&1
fi
log "⑧ H3 全臂统计..."
PYTHONPATH=code "$PY" code/tests/h3_charge_legality.py \
  --gen-root "$OUT/ligand" --ref-root "$OUT/ref" \
  --native-root "$TMROOT/ref_native" --uncond-root "$TMROOT/uncond" \
  --pH 7.4 --out output/h3_ligand_v14.json

# ============ ⑨ Tm/Sol（in 组）============
mkdir -p "$TMROOT/seqs"
for p in $(ls "$OUT/ligand" 2>/dev/null); do
  for arm in native n2 p2 n8 p8; do
    mkdir -p "$TMROOT/seqs/$p/arm_$arm"
    ln -sf "$OUT/ligand/$p/pH7.4/arm_$arm/seqs.fa" "$TMROOT/seqs/$p/arm_$arm/seqs.fa" 2>/dev/null
  done
done
log "⑨ Tm 预测（temberture，CPU）..."
HF_HUB_OFFLINE=1 timeout 28800 "$PYT" code/tests/temberture_score.py --input-dir "$TMROOT/seqs" \
  > log/v14_tm_seqs.log 2>&1
HF_HUB_OFFLINE=1 timeout 10800 "$PYT" code/tests/temberture_score.py --input-dir "$TMROOT/uncond" \
  > log/v14_tm_uncond.log 2>&1
log "⑨ protein-sol..."
for fa in $(find "$OUT/ligand" "$TMROOT/ref_native" "$TMROOT/uncond" \( -name "seqs.fa" -o -name "*_native.fa" \)); do
  python3 protein_sol_mcp/scripts/protein_sol_predict.py "$fa" > /dev/null 2>&1
done
log "⑨ Tm/Sol 汇总..."
PYTHONPATH=code "$PY" code/tests/ligand_v9/v12_2_ligand_tm_sol_summarize.py \
  --gen-root "$OUT/ligand" --tm-seqs-root "$TMROOT/seqs" \
  --ref-native-root "$TMROOT/ref_native" --uncond-root "$TMROOT/uncond" \
  --out "$TMROOT/tm_sol_summary.json"

# ============ boundary 1A65（单列，global 校准回退）============
log "★ boundary 1A65 采样（n50，单列）..."
PYTHONPATH=code timeout 7200 "$PY" code/tests/ligand_v9/validate_generalization.py \
  --manifest "$MAN_BD" --out_dir "$OUT" --mode ligand --backbone auto \
  --cond_encoder "$ENC" --weights "$W" \
  --n 50 --device "$SAMP_GPU" --pH 7.4 \
  --calibrate auto --calibration_file "$CAL" \
  > log/v14_val_sample_boundary.log 2>&1

touch log/v14_ligand_validation.DONE
log "== v14 验证链全流程完成 =="

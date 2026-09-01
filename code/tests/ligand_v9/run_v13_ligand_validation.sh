#!/usr/bin/env bash
# v13 配体删减根治复验链（PROJECT_LOCAL_V12_2 §7.7，判据修正：泛化 n50、H3 全臂）
#
# 环节：① 配体诊断 slope → ② 校准表 → ③ 泛化采样 n50（per-protein 校准）
#       → ④ 组成分析（带电总数 0.7-1.3×）→ ⑤ H1 折叠(ESMFold+TM) → ⑥ H2 电荷统计
#       → ⑦ H4 PROPKA → ⑧ H3 全臂复测 → ⑨ Tm/Sol 复测。断点续跑（按产物计数）。
#
# 判定（§4 + §7.7）：组成 0.7-1.3×、slope 校准后 [0.9,1.15]、H2 ≥ 72%、H1 TM≥0.7、
#   H3 ≤ max(native,uncond)+5pp（全臂 × n50，不只 n8）、Tm/Sol S2 无恶化。
#
# 用法（训练完成后）：nohup bash code/tests/ligand_v9/run_v13_ligand_validation.sh \
#   > log/v13_ligand_validation.stdout 2>&1 &
set -u
export PATH="/home/baokun_yu/miniconda3/envs/confumpnn/bin:$PATH"
ROOT="/data/nfs/IC/baokun_yu/ConfuMPNN"
cd "$ROOT"
PY="/home/baokun_yu/miniconda3/envs/confumpnn/bin/python"
PYE="/home/baokun_yu/miniconda3/envs/confumpnn-esmfold/bin/python"
PYT="/home/baokun_yu/miniconda3/envs/confumpnn-temberture/bin/python"   # Tm 专用环境
MANIFEST="data/validation_pdbs/validation_manifest.json"
ENC="output/finetune_ligand_v13/finetune_epoch030.pt"
W="LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt"
OUT="output/generalization_ligand_v13"
TMROOT="output/tm_sol_ligand_v13"
DIAG="output/v13_ligand_diag_response.json"
CAL="output/charge_calibration_v13_ligand.json"
TARGET=50   # 10 蛋白 × 5 臂

log() { echo "[$(date '+%H:%M:%S')] $1"; }
count() { find "$1" -name "$2" 2>/dev/null | wc -l; }

[ -f "$ENC" ] || { log "❌ checkpoint 缺失 $ENC（训练未完成？）"; exit 1; }

# ============ ① 配体诊断 slope（17 蛋白，targets 与 v12.2 配体一致）============
if [ ! -f "$DIAG" ]; then
  log "① 配体诊断启动（GPU6）..."
  PYTHONPATH=code timeout 14400 "$PY" index/v10_repair/v10_diag_response_curve.py \
    --backbone ligand_mpnn --cond_encoder "$ENC" --weights "$W" \
    --pdb-list log/v12_2_ligand_trainish.list --manifest "$MANIFEST" \
    --targets=-34,-30,-25,-20,-15,-10,-5,0,5,10,18 --include_native --n 20 \
    --out "$DIAG" > log/v13_ligand_diag.log 2>&1
else
  log "① 诊断已有，跳过"
fi

# ============ ② 校准表 ============
if [ ! -f "$CAL" ]; then
  log "② 建校准表..."
  PYTHONPATH=code "$PY" index/v10_repair/build_calibration.py \
    --diag "$DIAG" --label v13_ligand --out "$CAL"
fi

# ============ ③ 泛化采样 n50（配体模式 per-protein 校准）============
if [ "$(count "$OUT/ligand" validation.json)" -lt 10 ]; then
  log "③ 泛化采样 n50（GPU6）..."
  PYTHONPATH=code timeout 28800 "$PY" code/tests/ligand_v9/validate_generalization.py \
    --manifest "$MANIFEST" --out_dir "$OUT" --mode ligand --backbone auto \
    --cond_encoder "$ENC" --weights "$W" \
    --n 50 --device cuda:6 --pH 7.4 \
    --calibrate auto --calibration_file "$CAL" \
    > log/v13_val_sample.log 2>&1
fi
log "③ 采样完成：$(count "$OUT/ligand" validation.json)/10"

# ============ ④ 组成分析（判据 0.7-1.3×）============
log "④ 组成分析..."
PYTHONPATH=code "$PY" code/tests/ligand_v9/compare_comp_ligand.py \
  --gen-root "$OUT/ligand" --out output/v13_ligand_comp.json

# ============ ⑤ H1 折叠：ESMFold 回折 + TM-score ============
if [ "$(count "$OUT/ligand" plddt.csv)" -lt "$TARGET" ]; then
  log "⑤ 回折（GPU4）..."
  timeout 21600 "$PYE" code/tests/esmfold_score.py --input-dir "$OUT/ligand" --device cuda:4 \
    > log/v13_val_esmfold.log 2>&1
fi
log "⑤ TM-score..."
for arm_dir in $(find "$OUT/ligand" -type d -name "arm_*" 2>/dev/null); do
  [ -f "$arm_dir/tm.csv" ] && continue
  rel="${arm_dir#"$OUT"/ligand/}"
  pdb="$(echo "$rel" | cut -d/ -f1)"
  [ ! -d "$arm_dir/folds" ] && { log "⚠️ $rel folds 缺失"; continue; }
  timeout 600 "$PY" code/tests/tm_score.py --folds "$arm_dir/folds" \
    --ref "$OUT/ref/${pdb}_ref.pdb" --out "$arm_dir/tm.csv" \
    >> log/v13_val_tm.log 2>&1
done

# ============ ⑥ H2 统计 ============
log "⑥ 统计（H2/GRAVY/RMSD）..."
PYTHONPATH=code "$PY" code/tests/ligand_v9/generalization_stats.py \
  --root "$OUT/ligand" --manifest "$MANIFEST" --out output/v13_ligand_gen_stats.json \
  > log/v13_val_stats.log 2>&1

# ============ ⑦ H4 PROPKA（4 蛋白 × native/n8）============
log "⑦ PROPKA 复核..."
PYTHONPATH=code "$PY" - <<'PYEOF'
import json, os, subprocess
OUT = 'output/generalization_ligand_v13/ligand'
PY = '/home/baokun_yu/miniconda3/envs/confumpnn/bin/python'
PROTS = ['1BJ4', '1A65', '1AG0', '1C6O']
outdir = 'output/propka_v13_ligand'
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

# ============ ⑧ H3 全臂复测（判据 §7.7：全臂 × n50，不只 n8）============
# 前置：ref_native（从泛化 arm_native 提取）+ uncond（v13 采样）
mkdir -p "$TMROOT/ref_native" "$TMROOT/uncond"
PYTHONPATH=code "$PY" - <<'PYEOF'
import json
from pathlib import Path
ROOT = Path('/data/nfs/IC/baokun_yu/ConfuMPNN')
OUT = ROOT / 'output/generalization_ligand_v13/ligand'
REF = ROOT / 'output/tm_sol_ligand_v13/ref_native'
man = json.load(open(ROOT / 'data/validation_pdbs/validation_manifest.json'))
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
  log "⑧ uncond 采样（GPU6）..."
  PYTHONPATH=code timeout 14400 "$PY" code/tests/ligand_v9/sample_unconditioned_ligand.py \
    --manifest "$MANIFEST" --out_dir "$TMROOT/uncond" \
    --cond_encoder "$ENC" --weights "$W" \
    --n 30 --device cuda:6 --pH 7.4 > log/v13_uncond_sample.log 2>&1
fi
log "⑧ H3 全臂统计..."
PYTHONPATH=code "$PY" code/tests/h3_charge_legality.py \
  --gen-root "$OUT/ligand" --ref-root "$OUT/ref" \
  --native-root "$TMROOT/ref_native" --uncond-root "$TMROOT/uncond" \
  --pH 7.4 --out output/h3_ligand_v13.json

# ============ ⑨ Tm/Sol 复测（同批 n50 序列）============
mkdir -p "$TMROOT/seqs"
for p in $(ls "$OUT/ligand" 2>/dev/null); do
  for arm in native n2 p2 n8 p8; do
    mkdir -p "$TMROOT/seqs/$p/arm_$arm"
    ln -sf "$OUT/ligand/$p/pH7.4/arm_$arm/seqs.fa" "$TMROOT/seqs/$p/arm_$arm/seqs.fa"
  done
done
log "⑨ Tm 预测（confumpnn-temberture，CPU）..."
HF_HUB_OFFLINE=1 timeout 21600 "$PYT" code/tests/temberture_score.py --input-dir "$TMROOT/seqs" \
  > log/v13_tm_seqs.log 2>&1
HF_HUB_OFFLINE=1 timeout 7200 "$PYT" code/tests/temberture_score.py --input-dir "$TMROOT/uncond" \
  > log/v13_tm_uncond.log 2>&1
log "⑨ protein-sol（串行，共享 input.fasta）..."
for fa in $(find "$OUT/ligand" "$TMROOT/ref_native" "$TMROOT/uncond" -name "seqs.fa" -o -name "*_native.fa"); do
  python3 protein_sol_mcp/scripts/protein_sol_predict.py "$fa" > /dev/null 2>&1
done
log "⑨ Tm/Sol 汇总..."
PYTHONPATH=code "$PY" code/tests/ligand_v9/v12_2_ligand_tm_sol_summarize.py \
  --gen-root "$OUT/ligand" --tm-seqs-root "$TMROOT/seqs" \
  --ref-native-root "$TMROOT/ref_native" --uncond-root "$TMROOT/uncond" \
  --out "$TMROOT/tm_sol_summary.json"

touch log/v13_ligand_validation.DONE
log "== v13 复验链全流程完成 =="

#!/usr/bin/env bash
# v12.3 校准版泛化验证（MoMPNN protein 模式，9 单体含新增长蛋白）：采样 → ESMFold → TM → 统计 → PROPKA(H4)。
#
# 背景：v12.3 = v12.2 补 S40 长蛋白域重训（40ep，data 6580 域，L>400 8.8%），目标治长蛋白 OOD
#   （v12.2 验证 1A65/504、1BJ4/470 长蛋白 H2 fail）。本脚本验证 v12.3 + 新校准表（charge_calibration_v12_3.json）
#   的完整有效性：H2 电荷、H1 折叠、GRAVY、H4 PROPKA。
# 协议：9 蛋白（1AZM/1AS2/2FEO/5CQH/1CGE/1A65/1BJ4/13BB/1CDG，单链单体）× 5 臂（native/n2/p2/n8/p8）× n30 × pH7.4。
# 产物根：output/generalization_v12_3_calib
# 用法：nohup bash code/tests/ligand_v9/run_v12_3_validation.sh > log/v12_3_validation.stdout 2>&1 &
set -u
export PATH="/home/baokun_yu/miniconda3/envs/confumpnn/bin:$PATH"
ROOT="/data/nfs/IC/baokun_yu/ConfuMPNN"
cd "$ROOT"
PY="/home/baokun_yu/miniconda3/envs/confumpnn/bin/python"
PYE="/home/baokun_yu/miniconda3/envs/confumpnn-esmfold/bin/python"
MANIFEST="data/validation_pdbs/validation_manifest_v12_3.json"
ENC="output/finetune_v12_3/finetune_epoch040.pt"
W="MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt"
CAL="output/charge_calibration_v12_3.json"
OUT="output/generalization_v12_3_calib"
NPROT=9
TARGET=45   # 9 蛋白 × 5 臂

log() { echo "[$(date '+%H:%M:%S')] $1"; }
count() { find "$1" -name "$2" 2>/dev/null | wc -l; }

# ================= V1: 采样（protein 模式 5 臂 n30，校准）=================
log "== V1 采样 =="
if [ "$(count "$OUT" validation.json)" -lt "$NPROT" ]; then
  log "V1 采样启动（protein 模式 5 臂 n=30，校准，cuda:6）..."
  PYTHONPATH=code timeout 14400 "$PY" code/tests/ligand_v9/validate_generalization.py \
    --manifest "$MANIFEST" --out_dir "$OUT" --mode protein --backbone auto \
    --cond_encoder "$ENC" --weights "$W" \
    --protein_arms native,n2,p2,n8,p8 \
    --n 30 --device cuda:6 --pH 7.4 \
    --calibrate auto --calibration_file "$CAL" \
    > log/v12_3_val_sample.log 2>&1
else
  log "V1 采样已完成（$(count "$OUT" validation.json)/$NPROT），跳过"
fi
log "V1 采样完成：$(count "$OUT" validation.json)/$NPROT"

# ================= V2: ESMFold 回折（GPU6，勿 GPU4=v14）=================
log "== V2 ESMFold 回折 =="
if [ "$(count "$OUT" plddt.csv)" -lt "$TARGET" ]; then
  log "V2 回折启动（cuda:6）..."
  timeout 21600 "$PYE" code/tests/esmfold_score.py --input-dir "$OUT" --device cuda:6 \
    > log/v12_3_val_esmfold.log 2>&1
else
  log "V2 回折已完成（$(count "$OUT" plddt.csv)/$TARGET），跳过"
fi
log "V2 回折完成：$(count "$OUT" plddt.csv)/$TARGET"

# ================= V3: TM-score（US-align vs ref 骨架）=================
log "== V3 TM-score =="
for arm_dir in $(find "$OUT" -type d -name "arm_*" 2>/dev/null); do
  [ -f "$arm_dir/tm.csv" ] && continue
  rel="${arm_dir#"$OUT"/}"
  pdb="$(echo "$rel" | cut -d/ -f2)"
  if [ ! -d "$arm_dir/folds" ]; then
    log "V3 ⚠️ $rel folds 缺失（回折未完成），跳过"
    continue
  fi
  timeout 600 "$PY" code/tests/tm_score.py --folds "$arm_dir/folds" \
    --ref "$OUT/ref/${pdb}_ref.pdb" --out "$arm_dir/tm.csv" \
    >> log/v12_3_val_tm.log 2>&1
done
log "V3 TM 完成：$(count "$OUT" tm.csv)/$TARGET"

# ================= V4: 汇总统计 =================
log "== V4 统计 =="
PYTHONPATH=code "$PY" code/tests/ligand_v9/generalization_stats.py \
  --root "$OUT" --manifest "$MANIFEST" --out output/generalization_v12_3_calib_stats.json \
  > log/v12_3_val_stats.log 2>&1
log "V4 统计完成：output/generalization_v12_3_calib_stats.json"

# ================= V5: PROPKA 物理复核（H4，4 长蛋白 native/n8）=================
log "== V5 PROPKA 复核（4 长蛋白 × native/n8）=="
PYTHONPATH=code "$PY" - <<'PYEOF'
import json, subprocess, os
ROOT = 'output/generalization_v12_3_calib'
PY = '/home/baokun_yu/miniconda3/envs/confumpnn/bin/python'
PROTS = ['1BJ4', '1A65', '13BB', '1CDG']   # 长蛋白（v12.3 核心关注）
outdir = 'output/propka_v12_3'
os.makedirs(outdir, exist_ok=True)
for p in PROTS:
    vj_path = f'{ROOT}/protein/{p}/validation.json'
    if not os.path.exists(vj_path):
        print('SKIP (无 validation.json):', p); continue
    q = json.load(open(vj_path))['native_charge']
    for a, tag in [('native', q), ('n8', q - 8)]:
        folds = f'{ROOT}/protein/{p}/pH7.4/arm_{a}/folds'
        if not os.path.isdir(folds):
            print('SKIP (无 folds):', p, a); continue
        out = f'{outdir}/{p}_{a}.json'
        if os.path.exists(out):
            print('已有:', out); continue
        subprocess.run([PY, 'code/tests/propka_charge_check.py', '--pdb', folds,
                        '--pH', '7.4', '--target', str(round(tag)), '--out', out],
                       check=False)
        print('done:', out)
PYEOF
log "V5 PROPKA 完成"
touch log/v12_3_validation.DONE
log "== v12.3 泛化验证全流程完成（H1/H2/GRAVY/H4 产物齐备）=="

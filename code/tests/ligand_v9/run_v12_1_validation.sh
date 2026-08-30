#!/usr/bin/env bash
# v12.1 校准版泛化验证：采样 → ESMFold 回折 → TM-score → 统计 → PROPKA(H4)。
#
# 背景：v12.1（组成双计数 frac_floor0.5 + GRAVY margin0.4 + λ_v12 0.2）闭环诊断
#   slope 未达标(1.82)但组成健康（D/K 接近 native）→ 校准表 charge_calibration_v12_1.json
#   把 slope 修到 1.04。本脚本在泛化 10 蛋白上验证 v12.1+校准的完整有效性：
#   H2 电荷（dev≤2 达标率）、H1 折叠（TM 中位/≥0.7 率/失败率/RMSD）、GRAVY、H4 PROPKA 物理复核。
# 协议：10 蛋白 × 5 臂（native/n2/p2/n8/p8）× n=30 × pH 7.4，MoMPNN protein 模式。
#
# 用法（后台）：nohup bash code/tests/ligand_v9/run_v12_1_validation.sh > log/v12_1_validation.stdout 2>&1 &
# 断点续跑：按产物计数（validation.json / plddt.csv / tm.csv）判断阶段是否完成。
set -u
export PATH="/home/baokun_yu/miniconda3/envs/confumpnn/bin:$PATH"   # nohup 环境无 PATH，propka3 等 CLI 依赖它
ROOT="/data/nfs/IC/baokun_yu/ConfuMPNN"
cd "$ROOT"
PY="/home/baokun_yu/miniconda3/envs/confumpnn/bin/python"
PYE="/home/baokun_yu/miniconda3/envs/confumpnn-esmfold/bin/python"
MANIFEST="data/validation_pdbs/validation_manifest.json"
ENC="output/finetune_v12_1/finetune_epoch030.pt"
W="MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt"
CAL="output/charge_calibration_v12_1_n50.json"
OUT="output/generalization_v12_1_calib_n50"
TARGET=50   # 10 蛋白 × 5 臂

log() { echo "[$(date '+%H:%M:%S')] $1"; }
count() { find "$1" -name "$2" 2>/dev/null | wc -l; }

# ================= V1: 采样（protein 模式 5 臂 n30，校准，GPU6）=================
# 注意：validation.json 每蛋白 1 个（共 10 个），不能用 TARGET=50（那是 arm 数）
log "== V1 采样 =="
if [ "$(count "$OUT" validation.json)" -lt 10 ]; then
  log "V1 采样启动（protein 模式 5 臂 n=30，校准，cuda:6）..."
  PYTHONPATH=code timeout 14400 "$PY" code/tests/ligand_v9/validate_generalization.py \
    --manifest "$MANIFEST" --out_dir "$OUT" --mode protein --backbone auto \
    --cond_encoder "$ENC" --weights "$W" \
    --protein_arms native,n2,p2,n8,p8 \
    --n 30 --device cuda:6 --pH 7.4 \
    --calibrate auto --calibration_file "$CAL" \
    > log/v12_1_val_sample.log 2>&1
else
  log "V1 采样已完成（$(count "$OUT" validation.json)/$TARGET），跳过"
fi
log "V1 采样完成：$(count "$OUT" validation.json)/$TARGET"

# ================= V2: ESMFold 回折（GPU4）=================
log "== V2 ESMFold 回折 =="
if [ "$(count "$OUT" plddt.csv)" -lt "$TARGET" ]; then
  log "V2 回折启动（cuda:4）..."
  timeout 21600 "$PYE" code/tests/esmfold_score.py --input-dir "$OUT" --device cuda:4 \
    > log/v12_1_val_esmfold.log 2>&1
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
    >> log/v12_1_val_tm.log 2>&1
done
log "V3 TM 完成：$(count "$OUT" tm.csv)/$TARGET"

# ================= V4: 汇总统计 =================
log "== V4 统计 =="
PYTHONPATH=code "$PY" code/tests/ligand_v9/generalization_stats.py \
  --root "$OUT" --manifest "$MANIFEST" --out output/generalization_v12_1_calib_stats.json \
  > log/v12_1_val_stats.log 2>&1
log "V4 统计完成：output/generalization_v12_1_calib_stats.json"

# ================= V5: PROPKA 物理复核（H4）=================
log "== V5 PROPKA 复核（4 蛋白 × native/n8）=="
PYTHONPATH=code "$PY" - <<'PYEOF'
import json, subprocess, os
ROOT = 'output/generalization_v12_1_calib_n50'
PY = '/home/baokun_yu/miniconda3/envs/confumpnn/bin/python'
PROTS = ['1BJ4', '1A65', '1AG0', '1C6O']   # 覆盖最坏响应/长蛋白/中等
outdir = 'output/propka_v12_1_n50'
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
touch log/v12_1_validation.DONE
log "== v12.1 泛化验证全流程完成（H1/H2/GRAVY/H4 产物齐备）=="

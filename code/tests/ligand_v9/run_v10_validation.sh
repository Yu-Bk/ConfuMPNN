#!/usr/bin/env bash
# v10 双编码器泛化验证编排：采样 → ESMFold 回折 → TM-score → 统计。
#
# 背景：v10（A 条件解耦 + B 表面电荷监督 + C 结构惩罚）双 backbone 训练完成，
# 本脚本在泛化 10 蛋白上验证 v10 有效性：
#   H2 电荷控制（dev≤2.0 达标率）、H1 折叠（ESMFold 回折 TM 中位 / 失败率 / RMSD）、
#   pLDDT、%sol、Tm、GRAVY、多样性。
# 协议对齐 PROJECT_LOCAL.md §4：10 蛋白 × 5 臂（native/n2/p2/n8/p8）× n=30 × pH 7.4。
# MoMPNN（protein 模式，纯骨架）+ LigandMPNN（both 模式，配体上下文）并行。
#
# 用法（后台）：
#   nohup bash code/tests/ligand_v9/run_v10_validation.sh > log/v10_validation.stdout 2>&1 &
#
# 断点续跑：按产物计数（validation.json / plddt.csv / tm.csv）判断阶段是否完成，跳过已完成。
#
# 产物：
#   output/generalization_v10_mompnn/  +  output/generalization_v10_ligand/
#   output/generalization_v10_{mompnn,ligand}_stats.json
set -u

ROOT="/data/nfs/IC/baokun_yu/ConfuMPNN"
cd "$ROOT"
PY="/home/baokun_yu/miniconda3/envs/confumpnn/bin/python"
PYE="/home/baokun_yu/miniconda3/envs/confumpnn-esmfold/bin/python"
MANIFEST="data/validation_pdbs/validation_manifest.json"
OUT_M="output/generalization_v10_mompnn"
OUT_L="output/generalization_v10_ligand"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

count() { find "$1" -name "$2" 2>/dev/null | wc -l; }

TARGET=50   # 10 蛋白 × 5 臂 = 50 个 arm

# ================= V1: 采样（两编码器并行）=================
# MoMPNN：protein 模式，显式传 5 臂（默认仅 3 臂 native/n8/p8，需对齐协议）
log "== V1 采样 =="
if [ "$(count "$OUT_M" validation.json)" -lt "$TARGET" ]; then
  log "V1[MoMPNN] 启动采样（protein 模式，5 臂，n=30，cuda:0）..."
  PYTHONPATH=code timeout 14400 "$PY" code/tests/ligand_v9/validate_generalization.py \
    --manifest "$MANIFEST" --out_dir "$OUT_M" --mode protein --backbone auto \
    --cond_encoder output/finetune_v10_mompnn/finetune_epoch030.pt \
    --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt \
    --protein_arms native,n2,p2,n8,p8 \
    --n 30 --device cuda:1 --pH 7.4 \
    > log/v10_val_mompnn_sample.log 2>&1 &
  SAMPLING_M=yes
else
  log "V1[MoMPNN] 采样已完成（$(count "$OUT_M" validation.json)/$TARGET），跳过"
fi

if [ "$(count "$OUT_L" validation.json)" -lt "$TARGET" ]; then
  log "V1[LigandMPNN] 启动采样（both 模式，5 臂，n=30，cuda:1）..."
  PYTHONPATH=code timeout 14400 "$PY" code/tests/ligand_v9/validate_generalization.py \
    --manifest "$MANIFEST" --out_dir "$OUT_L" --mode both --backbone auto \
    --cond_encoder output/finetune_v10_ligand/finetune_epoch030.pt \
    --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
    --n 30 --device cuda:2 --pH 7.4 \
    > log/v10_val_ligand_sample.log 2>&1 &
  SAMPLING_L=yes
else
  log "V1[LigandMPNN] 采样已完成（$(count "$OUT_L" validation.json)/$TARGET），跳过"
fi
wait   # 等两个采样都完成（含 timeout 触发的情形）
log "V1 采样完成：MoMPNN=$(count "$OUT_M" validation.json)/$TARGET，LigandMPNN=$(count "$OUT_L" validation.json)/$TARGET"

# ================= V2: ESMFold 回折（两编码器并行）=================
# esmfold_score.py 批量模式：递归所有 seqs.fa，同目录写 plddt.csv + folds/（回折 PDB）
log "== V2 ESMFold 回折 =="
if [ "$(count "$OUT_M" plddt.csv)" -lt "$TARGET" ]; then
  log "V2[MoMPNN] 回折启动（cuda:0）..."
  timeout 21600 "$PYE" code/tests/esmfold_score.py --input-dir "$OUT_M" --device cuda:1 \
    > log/v10_val_mompnn_esmfold.log 2>&1 &
else
  log "V2[MoMPNN] 回折已完成（$(count "$OUT_M" plddt.csv)/$TARGET），跳过"
fi
if [ "$(count "$OUT_L" plddt.csv)" -lt "$TARGET" ]; then
  log "V2[LigandMPNN] 回折启动（cuda:1）..."
  timeout 21600 "$PYE" code/tests/esmfold_score.py --input-dir "$OUT_L" --device cuda:2 \
    > log/v10_val_ligand_esmfold.log 2>&1 &
else
  log "V2[LigandMPNN] 回折已完成（$(count "$OUT_L" plddt.csv)/$TARGET），跳过"
fi
wait
log "V2 回折完成：MoMPNN=$(count "$OUT_M" plddt.csv)/$TARGET，LigandMPNN=$(count "$OUT_L" plddt.csv)/$TARGET"

# ================= V3: TM-score（US-align vs ref 骨架）=================
# 对每个 arm 目录：tm_score.py --folds <arm>/folds --ref <out>/ref/{pdb}_ref.pdb --out <arm>/tm.csv
# ref 由 validate_generalization.py 自动生成（N,CA,C 骨架）
log "== V3 TM-score =="
run_tm() {
  local out=$1
  for arm_dir in $(find "$out" -type d -name "arm_*" 2>/dev/null); do
    if [ -f "$arm_dir/tm.csv" ]; then
      continue
    fi
    local rel="${arm_dir#"$out"/}"     # {mode}/{pdb}/pH7.4/arm_{tag}
    local pdb="$(echo "$rel" | cut -d/ -f2)"
    if [ ! -d "$arm_dir/folds" ]; then
      log "V3[$out] ⚠️ $arm_dir/folds 缺失（回折未完成），跳过"
      continue
    fi
    log "V3[$out] TM: $rel"
    timeout 600 "$PY" code/tests/tm_score.py \
      --folds "$arm_dir/folds" \
      --ref "$out/ref/${pdb}_ref.pdb" \
      --out "$arm_dir/tm.csv"
  done
}
run_tm "$OUT_M"
run_tm "$OUT_L"
log "V3 TM 完成：MoMPNN=$(count "$OUT_M" tm.csv)/$TARGET，LigandMPNN=$(count "$OUT_L" tm.csv)/$TARGET"

# ================= V4: 汇总统计 =================
log "== V4 统计 =="
PYTHONPATH=code "$PY" code/tests/ligand_v9/generalization_stats.py \
  --root "$OUT_M" --manifest "$MANIFEST" --out output/generalization_v10_mompnn_stats.json \
  > log/v10_val_mompnn_stats.log 2>&1
PYTHONPATH=code "$PY" code/tests/ligand_v9/generalization_stats.py \
  --root "$OUT_L" --manifest "$MANIFEST" --out output/generalization_v10_ligand_stats.json \
  > log/v10_val_ligand_stats.log 2>&1
log "== v10 泛化验证全流程完成：output/generalization_v10_{mompnn,ligand}_stats.json =="
touch log/v10_validation.DONE

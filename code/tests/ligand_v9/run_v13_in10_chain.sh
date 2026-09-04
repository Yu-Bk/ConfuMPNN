#!/usr/bin/env bash
# v13 配体【in-10 全链权威对照】—— v14-clean 同一套测试协议（新 in-10 manifest + v13 自己的校准）
#
# 背景：v13 旧对照用的是旧测试集（output/generalization_ligand_v13 等，已归档/保留不动）。
#       为了与 v14-clean 同集同协议可比，用 v14 的 in-10 manifest + boundary 1A65，把 v13 完整重跑。
# v13 特点：encoder = finetune_ligand_v13/finetune_epoch030.pt（训练无 RNA/DNA、A1 pocket 非 global）；
#           backbone = ligandmpnn_v_32_010_25.pt（num_ligand_atoms=25）。
# RNA/DNA 成员（5O60_E/21KL_A/9DWG_L/3MXB_A）对 v13 是 out-of-domain → 预期偏弱，作"数据扩充收益"论据。
#
# 环节：① diag slope → ② 校准表 → ③ n50 采样（in 10×5臂）→ ④ 组成 → ⑤ ESMFold H1 + TM
#       → ⑥ H2 统计 → ⑦ PROPKA H4 → ⑧ ref_native/uncond + H3 → ⑨ Tm/Sol S2 → ★ boundary 1A65
# 每步产物存在性检查 → 只补缺失阶段；全链成功才 touch DONE（杜绝假完成）。
#
# 用法：nohup bash code/tests/ligand_v9/run_v13_in10_chain.sh > log/v13_in10_chain.stdout 2>&1 &
set -u
export PATH="/home/baokun_yu/miniconda3/envs/confumpnn/bin:$PATH"
ROOT="/data/nfs/IC/baokun_yu/ConfuMPNN"
cd "$ROOT"
PY="/home/baokun_yu/miniconda3/envs/confumpnn/bin/python"
PYE="/home/baokun_yu/miniconda3/envs/confumpnn-esmfold/bin/python"
PYT="/home/baokun_yu/miniconda3/envs/confumpnn-temberture/bin/python"
DIAG_GPU="${DIAG_GPU:-cuda:6}"   # GPU6：Task2(fixbinding) 已完成转 CPU，可独占
SAMP_GPU="${SAMP_GPU:-cuda:6}"
ESM_GPU="${ESM_GPU:-cuda:6}"
MAN_IN="data/validation_pdbs/validation_manifest_v14_in.json"    # 新 in-10（含 5O60_E，无 2E9R_X）
MAN_BD="data/validation_pdbs/validation_manifest_v14_boundary.json"  # 1A65
ENC="output/finetune_ligand_v13/finetune_epoch030.pt"
W="LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt"
NATOMS=25
# ---- 全新 in10 命名（与 v14 *_clean、旧 v13 均隔离）----
OUT="output/generalization_ligand_v13_in10"
TMROOT="output/tm_sol_ligand_v13_in10"
DIAG="output/v13_ligand_diag_response_in10.json"
CAL="output/charge_calibration_v13_ligand_in10.json"
COMP="output/v13_ligand_comp_in10.json"
STATS="output/v13_ligand_gen_stats_in10.json"
H3OUT="output/h3_ligand_v13_in10.json"
PROPKA_DIR="output/propka_v13_ligand_in10"
mkdir -p "$OUT" "$TMROOT"

log() { echo "[$(date '+%H:%M:%S')] $1"; }
count() { find "$1" -name "$2" 2>/dev/null | wc -l; }
die()  { log "❌ $1"; log "== v13 in-10 链异常终止（未写 DONE）=="; exit 1; }

# ---- 输入预检 ----
for f in "$ENC" "$W" "$MAN_IN" "$MAN_BD" log/v14_ligand_trainish.list; do
  [ -f "$f" ] || die "输入缺失 $f"
done
grep -q "5O60_E" "$MAN_IN" || die "manifest 不含 5O60_E（非新 in-10）"
grep -q "1A65" "$MAN_BD" || die "boundary manifest 不含 1A65"

# ============ ① 配体诊断 slope（in 10 + trainish 8 = 18 蛋白）============
if [ ! -f "$DIAG" ]; then
  log "① 配体诊断启动（$DIAG_GPU）..."
  PYTHONPATH=code timeout 21600 "$PY" index/v10_repair/v10_diag_response_curve.py \
    --backbone ligand_mpnn --cond_encoder "$ENC" --weights "$W" \
    --pdb-list log/v14_ligand_trainish.list --manifest "$MAN_IN" \
    --targets=-34,-30,-25,-20,-15,-10,-5,0,5,10,18 --include_native --n 20 \
    --num_ligand_atoms $NATOMS \
    --device "$DIAG_GPU" --out "$DIAG" > log/v13_ligand_diag_in10.log 2>&1 \
    || die "① 诊断失败，见 log/v13_ligand_diag_in10.log"
  [ -f "$DIAG" ] || die "① 无产物 $DIAG"
  log "① 诊断完成：$(python3 -c "import json;d=json.load(open('$DIAG'));print(len(d.get('proteins',{})),'proteins')" 2>/dev/null)"
else
  log "① 诊断已存在，跳过"
fi

# ============ ② 校准表（v13 自己的，不借 v14 cal）============
if [ ! -f "$CAL" ]; then
  log "② 建校准表..."
  PYTHONPATH=code "$PY" index/v10_repair/build_calibration.py \
    --diag "$DIAG" --label v13_ligand_in10 --out "$CAL" \
    || die "② 校准失败"
  [ -f "$CAL" ] || die "② 无产物 $CAL"
  log "② 校准表完成"
else
  log "② 校准表已存在，跳过"
fi

# ============ ③ 泛化采样 n50（in 10 蛋白 × 5 臂，per-protein 校准）============
NVAL=$(count "$OUT/ligand" validation.json)
if [ "$NVAL" -lt 10 ]; then
  log "③ 泛化采样 n50（$SAMP_GPU）..."
  PYTHONPATH=code timeout 28800 "$PY" code/tests/ligand_v9/validate_generalization.py \
    --manifest "$MAN_IN" --out_dir "$OUT" --mode ligand --backbone auto \
    --cond_encoder "$ENC" --weights "$W" \
    --n 50 --device "$SAMP_GPU" --pH 7.4 \
    --calibrate auto --calibration_file "$CAL" \
    > log/v13_val_sample_in10.log 2>&1 || die "③ 采样失败，见 log/v13_val_sample_in10.log"
  NVAL=$(count "$OUT/ligand" validation.json)
  [ "$NVAL" -ge 10 ] || die "③ 采样未完成（$NVAL/10），见 log/v13_val_sample_in10.log"
  log "③ 采样完成：$NVAL/10"
else
  log "③ 采样已存在（$NVAL/10），跳过"
fi

# ============ ④ 组成分析（in-10，native 臂生成 vs native D/E/K/R）============
if [ ! -f "$COMP" ]; then
  log "④ 组成分析..."
  PYTHONPATH=code "$PY" code/tests/ligand_v9/compare_comp_ligand.py \
    --gen-root "$OUT/ligand" --manifest "$MAN_IN" --out "$COMP" \
    > log/v13_comp_in10.log 2>&1 || die "④ 组成失败，见 log/v13_comp_in10.log"
  [ -f "$COMP" ] || die "④ 无产物 $COMP"
  log "④ 组成完成"
else
  log "④ 组成已存在，跳过"
fi

# ============ ⑤ H1 折叠：ESMFold + TM-score ============
NPD=$(count "$OUT/ligand" plddt.csv)
if [ "$NPD" -lt 50 ]; then
  log "⑤ 回折（$ESM_GPU，预计 3-5h）..."
  timeout 21600 "$PYE" code/tests/esmfold_score.py --input-dir "$OUT/ligand" --device "$ESM_GPU" \
    > log/v13_val_esmfold_in10.log 2>&1 || die "⑤ ESMFold 失败，见 log/v13_val_esmfold_in10.log"
  NPD=$(count "$OUT/ligand" plddt.csv)
  [ "$NPD" -ge 50 ] || die "⑤ ESMFold 产出不足（$NPD/50）"
  log "⑤ ESMFold 完成（$NPD plddt）"
else
  log "⑤ ESMFold 产物已够（$NPD），跳过折叠"
fi
log "⑤ TM-score..."
NTM=$(count "$OUT/ligand" tm.csv)
if [ "$NTM" -lt 50 ]; then
  for arm_dir in $(find "$OUT/ligand" -type d -name "arm_*" 2>/dev/null); do
    [ -f "$arm_dir/tm.csv" ] && continue
    rel="${arm_dir#"$OUT"/ligand/}"
    pdb="$(echo "$rel" | cut -d/ -f1)"
    [ ! -d "$arm_dir/folds" ] && { log "⚠️ $rel folds 缺失，跳过 TM"; continue; }
    timeout 600 "$PY" code/tests/tm_score.py --folds "$arm_dir/folds" \
      --ref "$OUT/ref/${pdb}_ref.pdb" --out "$arm_dir/tm.csv" \
      >> log/v13_val_tm_in10.log 2>&1
  done
  NTM=$(count "$OUT/ligand" tm.csv)
  [ "$NTM" -ge 50 ] || die "⑤ TM-score 产出不足（$NTM/50），见 log/v13_val_tm_in10.log"
  log "⑤ TM-score 完成（$NTM/50）"
else
  log "⑤ TM-score 产物已够（$NTM），跳过"
fi

# ============ ⑥ H2 统计（含 TM/pLDDT/RMSD 联报）============
if [ ! -f "$STATS" ]; then
  log "⑥ 统计（H2/GRAVY/RMSD）..."
  PYTHONPATH=code "$PY" code/tests/ligand_v9/generalization_stats.py \
    --root "$OUT" --manifest "$MAN_IN" --out "$STATS" \
    > log/v13_val_stats_in10.log 2>&1 || die "⑥ 统计失败，见 log/v13_val_stats_in10.log"
  [ -f "$STATS" ] || die "⑥ 无产物 $STATS"
  log "⑥ 统计完成"
else
  log "⑥ 统计已存在，跳过"
fi

# ============ ⑦ PROPKA H4（in 代表：1BJ4/21KL_A/3MXB_A，native/n8）============
mkdir -p "$PROPKA_DIR"
NPROP=$(ls "$PROPKA_DIR"/*.json 2>/dev/null | wc -l)
if [ "$NPROP" -lt 6 ]; then
  log "⑦ PROPKA 复核..."
  for p in 1BJ4 21KL_A 3MXB_A; do
    vj="$OUT/ligand/$p/validation.json"
    [ -f "$vj" ] || { log "SKIP $p（无 validation.json）"; continue; }
    q=$("$PY" -c "import json;print(json.load(open('$vj'))['native_charge'])")
    for a in native n8; do
      case $a in native) tag=$q;; n8) tag=$(python3 -c "print(round($q-8))");; esac
      folds="$OUT/ligand/$p/pH7.4/arm_$a/folds"
      [ -d "$folds" ] || { log "SKIP folds $p $a"; continue; }
      out="$PROPKA_DIR/${p}_${a}.json"
      [ -f "$out" ] && continue
      PYTHONPATH=code "$PY" code/tests/propka_charge_check.py --pdb "$folds" \
        --pH 7.4 --target "$tag" --out "$out" || log "⚠️ PROPKA $p $a 异常"
    done
  done
  NPROP=$(ls "$PROPKA_DIR"/*.json 2>/dev/null | wc -l)
  [ "$NPROP" -ge 6 ] || die "⑦ PROPKA 产出不足（$NPROP/6）"
  log "⑦ PROPKA 完成（$NPROP 文件）"
else
  log "⑦ PROPKA 已存在（$NPROP 文件），跳过"
fi

# ============ ⑧ H3 全臂（in）：ref_native 抽取 + uncond 采样 + H3 统计 ============
mkdir -p "$TMROOT/ref_native" "$TMROOT/uncond"
NREF=$(ls "$TMROOT/ref_native"/*_native.fa 2>/dev/null | wc -l)
if [ "$NREF" -lt 10 ]; then
  log "⑧ ref_native 抽取..."
  PYTHONPATH=code "$PY" - <<PYEOF || die "⑧ ref_native 抽取失败"
import json
from pathlib import Path
ROOT = Path('$ROOT')
OUT = ROOT / '$OUT/ligand'
REF = ROOT / '$TMROOT/ref_native'
man = json.load(open(ROOT / '$MAN_IN'))
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
  NREF=$(ls "$TMROOT/ref_native"/*_native.fa 2>/dev/null | wc -l)
  [ "$NREF" -ge 10 ] || die "⑧ ref_native 不足（$NREF/10）"
  log "⑧ ref_native 完成（$NREF）"
else
  log "⑧ ref_native 已存在（$NREF），跳过"
fi
NUNC=$(find "$TMROOT/uncond" -name "*.fa" 2>/dev/null | wc -l)
if [ "$NUNC" -lt 10 ]; then
  log "⑧ uncond 采样（$SAMP_GPU）..."
  PYTHONPATH=code timeout 21600 "$PY" code/tests/ligand_v9/sample_unconditioned_ligand.py \
    --manifest "$MAN_IN" --out_dir "$TMROOT/uncond" \
    --cond_encoder "$ENC" --weights "$W" \
    --n 30 --device "$SAMP_GPU" --pH 7.4 > log/v13_uncond_sample_in10.log 2>&1 \
    || die "⑧ uncond 采样失败，见 log/v13_uncond_sample_in10.log"
  NUNC=$(find "$TMROOT/uncond" -name "*.fa" 2>/dev/null | wc -l)
  [ "$NUNC" -ge 10 ] || die "⑧ uncond 产物不足（$NUNC/10）"
  log "⑧ uncond 采样完成（$NUNC）"
else
  log "⑧ uncond 已存在（$NUNC），跳过"
fi
if [ ! -f "$H3OUT" ]; then
  log "⑧ H3 全臂统计..."
  PYTHONPATH=code "$PY" code/tests/h3_charge_legality.py \
    --gen-root "$OUT/ligand" --ref-root "$OUT/ref" \
    --native-root "$TMROOT/ref_native" --uncond-root "$TMROOT/uncond" \
    --manifest "$MAN_IN" --pH 7.4 --out "$H3OUT" \
    > log/v13_h3_in10.log 2>&1 || die "⑧ H3 统计失败，见 log/v13_h3_in10.log"
  [ -f "$H3OUT" ] || die "⑧ 无产物 $H3OUT"
  log "⑧ H3 完成：$(python3 -c "import json;r=json.load(open('$H3OUT'));print(r['pass'])" 2>/dev/null)"
else
  log "⑧ H3 已存在，跳过"
fi

# ============ ⑨ Tm/Sol（in 组）============
mkdir -p "$TMROOT/seqs"
# 绝对路径 symlink（相对路径会相对 symlink 自身目录解析 → 悬空）
GEN_ABS="$ROOT/$OUT/ligand"
for p in $(ls "$OUT/ligand" 2>/dev/null); do
  for arm in native n2 p2 n8 p8; do
    mkdir -p "$TMROOT/seqs/$p/arm_$arm"
    ln -sf "$GEN_ABS/$p/pH7.4/arm_$arm/seqs.fa" "$TMROOT/seqs/$p/arm_$arm/seqs.fa" 2>/dev/null
  done
done
NTMSEQ=$(find "$TMROOT/seqs" -name "*.tm.csv" 2>/dev/null | wc -l)
if [ "$NTMSEQ" -lt 50 ]; then
  log "⑨ Tm 预测（temberture，CPU，预计 1-3h）..."
  HF_HUB_OFFLINE=1 timeout 28800 "$PYT" code/tests/temberture_score.py --input-dir "$TMROOT/seqs" \
    > log/v13_tm_seqs_in10.log 2>&1 || die "⑨ temberture(seqs) 失败，见 log/v13_tm_seqs_in10.log"
  NTMSEQ=$(find "$TMROOT/seqs" -name "*.tm.csv" 2>/dev/null | wc -l)
  [ "$NTMSEQ" -ge 50 ] || die "⑨ temberture 产物不足（$NTMSEQ/50）"
  log "⑨ temberture(seqs) 完成（$NTMSEQ/50）"
else
  log "⑨ temberture(seqs) 产物已够（$NTMSEQ），跳过"
fi
# uncond Tm（尽量跑，不 gate 全链）
log "⑨ temberture(uncond)..."
HF_HUB_OFFLINE=1 timeout 10800 "$PYT" code/tests/temberture_score.py --input-dir "$TMROOT/uncond" \
  > log/v13_tm_uncond_in10.log 2>&1 || log "⚠️ temberture(uncond) 退出码非 0（可能部分）"
log "⑨ protein-sol..."
for fa in $(find "$OUT/ligand" "$TMROOT/ref_native" "$TMROOT/uncond" \( -name "seqs.fa" -o -name "*_native.fa" \) 2>/dev/null); do
  python3 protein_sol_mcp/scripts/protein_sol_predict.py "$fa" > /dev/null 2>&1
done
log "⑨ Tm/Sol 汇总..."
if [ ! -f "$TMROOT/tm_sol_summary.json" ] || [ ! -s "$TMROOT/tm_sol_summary.json" ]; then
  PYTHONPATH=code "$PY" code/tests/ligand_v9/v12_2_ligand_tm_sol_summarize.py \
    --gen-root "$OUT/ligand" --tm-seqs-root "$TMROOT/seqs" \
    --ref-native-root "$TMROOT/ref_native" --uncond-root "$TMROOT/uncond" \
    --manifest "$MAN_IN" --out "$TMROOT/tm_sol_summary.json" || die "⑨ 汇总失败"
  [ -s "$TMROOT/tm_sol_summary.json" ] || die "⑨ 汇总为空"
  log "⑨ Tm/Sol 完成"
else
  log "⑨ tm_sol_summary 已存在，跳过"
fi

# ============ ★ boundary 1A65（单列，global 校准回退，不进判据）============
if [ ! -f "$OUT/ligand/1A65/validation.json" ]; then
  log "★ boundary 1A65 采样（n50，$SAMP_GPU）..."
  PYTHONPATH=code timeout 7200 "$PY" code/tests/ligand_v9/validate_generalization.py \
    --manifest "$MAN_BD" --out_dir "$OUT" --mode ligand --backbone auto \
    --cond_encoder "$ENC" --weights "$W" \
    --n 50 --device "$SAMP_GPU" --pH 7.4 \
    --calibrate auto --calibration_file "$CAL" \
    > log/v13_val_sample_boundary_in10.log 2>&1 || die "★ boundary 1A65 采样失败"
  [ -f "$OUT/ligand/1A65/validation.json" ] || die "★ boundary 1A65 无产物"
  log "★ boundary 1A65 采样完成"
else
  log "★ boundary 1A65 已存在，跳过"
fi

touch log/v13_ligand_in10_chain.DONE
log "== v13 in-10 链全流程完成 =="

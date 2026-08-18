#!/usr/bin/env bash
# v9 验证：LigandMPNN 配体模式电荷控制修复检验。
#
# 背景：v7 编码器在 MoMPNN 上训练，LigandMPNN 配体模式电荷失效（1MBN dev 14.05）。
# v9 在 LigandMPNN backbone 上重训编码器 → 验证配体模式电荷是否恢复。
#
# 用法（任意目录，confumpnn 环境）：
#   bash code/tests/ligand_v9/validate_v9.sh [epoch]
# 默认用 output/finetune_ligand_v9/finetune_epoch030.pt；可传 epoch 号用中途 checkpoint。
#
# 验证蛋白：1MBN（+2 核心）、4DFR_chainA（-9）、1FQG（-6，假阳性对照）
# 输出：output/transfer_v9/{pdb}/pH7.4/seqs.fa + transfer_v9_stats.json
set -e
CONDA_DIR="${HOME}/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_DIR"
conda activate confumpnn
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"   # 项目根
cd "$ROOT"

EP="${1:-30}"
ENC="output/finetune_ligand_v9/finetune_epoch0${EP}.pt"
if [ ! -f "$ENC" ]; then ENC="output/finetune_ligand_v9/finetune_epoch${EP}.pt"; fi
if [ ! -f "$ENC" ]; then echo "❌ checkpoint 不存在: $ENC"; exit 1; fi
echo "=== v9 验证: encoder=$ENC (cuda:3) ==="

# 验证蛋白路径映射（1MBN/4DFR 用 transfer_test 单体，1FQG 用 ligand_test）
declare -A SRC=(
  [1MBN]="data/transfer_test/1MBN.pdb"
  [4DFR]="data/transfer_test/4DFR_chainA.pdb"
  [1FQG]="data/ligand_test/1FQG.pdb"
)
for pdb in 1MBN 4DFR 1FQG; do
  echo "--- ${pdb} ---"
  PYTHONPATH=code python code/tests/transfer_validation.py \
    --pdb "${SRC[$pdb]}" --mode ligand \
    --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
    --cond_encoder "$ENC" \
    --pH_list 7.4 \
    --out_dir output/transfer_v9 --n 20 --device cuda:3
done

echo "=== 汇总 ==="
PYTHONPATH=code python code/tests/transfer_stats.py \
  --root output/transfer_v9 --uncond_root output/transfer_uncond \
  --out output/transfer_v9_stats.json
echo "=== 完成: output/transfer_v9_stats.json ==="

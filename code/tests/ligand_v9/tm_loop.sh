#!/usr/bin/env bash
# v9 泛化验证 TM-score 批量计算：每臂回折结构 vs 参考骨架。
#
# ESMFold 回折完成后运行（esmfold_score.py 输出 folds/ + plddt.csv）。
# 用法（confumpnn 环境）：
#   bash code/tests/ligand_v9/tm_loop.sh [root_dir]
set -e
ROOT="${1:-output/generalization_v9}"
cd "$(dirname "$0")/../../.."

n=0
for arm_dir in $(find "$ROOT" -type d -name "arm_*" | sort); do
  pdb="$(echo "$arm_dir" | cut -d/ -f4)"
  ref="$ROOT/ref/${pdb}_ref.pdb"
  if [ ! -d "$arm_dir/folds" ]; then
    echo "⚠️  无 folds 目录，跳过: $arm_dir"
    continue
  fi
  echo "--- $pdb $(basename "$arm_dir") ---"
  conda run -n confumpnn python code/tests/tm_score.py \
    --folds "$arm_dir/folds" --ref "$ref" --out "$arm_dir/tm.csv" || true
  n=$((n+1))
done
echo "=== TM-score 完成 $n 臂 ==="

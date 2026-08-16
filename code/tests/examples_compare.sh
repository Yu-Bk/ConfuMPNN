#!/bin/bash
# 任务28：示例蛋白 × pH × 预设 生成对比（计划1 Phase 1 收尾）
# 对比维度：① 预设效果（pH7.4 × 4 预设）② pH 效果（default × 3 pH）
# 输出：code/output/examples/{pdb}_{preset}_pH{pH}/ + 汇总 log
source /home/baokun_yu/miniconda3/etc/profile.d/conda.sh
conda activate confumpnn
cd /data/nfs/IC/baokun_yu/ConfuMPNN/code
mkdir -p log

PROTEINS="1BC8 1UBQ 2LZM 1CRN"

for pdb in $PROTEINS; do
  # ① 预设对比（pH 7.4，target=0）
  for preset in default nucleic_acid_binding membrane acidic; do
    python run_guided.py --pdb input/${pdb}.pdb --pH 7.4 --target_charge 0 \
        --preset $preset --num_samples 5 --out_dir output/examples/${pdb}_${preset}_pH7.4 \
        > log/ex_${pdb}_${preset}.log 2>&1
  done
  # ② pH 对比（default 预设，无电荷引导以便隔离 pH 影响）
  for pH in 5.5 7.4 9.0; do
    python run_guided.py --pdb input/${pdb}.pdb --pH $pH --preset default \
        --num_samples 5 --out_dir output/examples/${pdb}_default_pH${pH} \
        > log/ex_${pdb}_pH${pH}.log 2>&1
  done
done
echo "ALL EXAMPLES DONE"

#!/bin/bash
# E1b 验证扩展：三目标打分 + TM-score 自洽性
# 前置：e1_extended.sh 采样完成（code/output/e1_ext/）
# 依赖环境：confumpnn(USalign) / confumpnn-esmfold / confumpnn-temberture / 系统 python(Protein-Sol)
source /home/baokun_yu/miniconda3/etc/profile.d/conda.sh
cd /data/nfs/IC/baokun_yu/ConfuMPNN
mkdir -p log

declare -A REF
REF[1BC8]=code/input/1BC8_chainC.pdb   # 三链复合物 → 用提取的纯蛋白链 C
REF[1CRN]=code/input/1CRN.pdb
REF[1UBQ]=code/input/1UBQ.pdb
REF[2LZM]=code/input/2LZM.pdb

echo "=== [1/4] ESMFold 回折 + pLDDT（confumpnn-esmfold）==="
conda run -n confumpnn-esmfold python code/tests/esmfold_score.py --input-dir code/output/e1_ext > log/e1ext_esmfold.log 2>&1
echo "  DONE (log/e1ext_esmfold.log)"

echo "=== [2/4] TM-score 自洽性（confumpnn / USalign）==="
for pdb in 1BC8 1CRN 1UBQ 2LZM; do
  ref=${REF[$pdb]}
  for model in ligand mompnn; do
    for cond in code/output/e1_ext/${pdb}_${model}/*/; do
      if [ -d "$cond/folds" ]; then
        cn=$(basename "$cond")
        conda run -n confumpnn python code/tests/tm_score.py --folds "$cond/folds" --ref "$ref" \
          --out "$cond/tm.csv" > log/tm_${pdb}_${model}_${cn}.log 2>&1
      fi
    done
  done
done
echo "  DONE"

echo "=== [3/4] Protein-Sol %sol（系统 python + Perl）==="
for fa in code/output/e1_ext/*/*/seqs.fa; do
  python3 protein_sol_mcp/scripts/protein_sol_predict.py "$fa" > log/protsol_$(basename "$(dirname "$fa")").log 2>&1
done
echo "  DONE"

echo "=== [4/4] TemBERTure Tm（confumpnn-temberture）==="
conda run -n confumpnn-temberture python code/tests/temberture_score.py --input-dir code/output/e1_ext > log/e1ext_temberture.log 2>&1
echo "  DONE (log/e1ext_temberture.log)"
echo "ALL SCORING DONE"

#!/bin/bash
# 阶段1：pH 响应对比 — MoMPNN vs 原版 LigandMPNN，5 组条件
source /home/baokun_yu/miniconda3/etc/profile.d/conda.sh
conda activate confumpnn
cd /data/nfs/IC/baokun_yu/ConfuMPNN/code

LIG=../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt
MOM=../MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt

for wname in ligand mompnn; do
  if [ "$wname" = "ligand" ]; then W=$LIG; else W=$MOM; fi
  for spec in 7.4_-8 7.4_0 7.4_8 4.0_0 10.0_0; do
    pH=${spec%%_*}; T=${spec##*_}
    out=output/compare/${wname}_pH${pH}_t${T}
    python run_guided.py --pdb input/1BC8.pdb --pH "$pH" --target_charge="$T" \
      --num_samples 10 --seed 111 --weights "$W" --out_dir "$out" \
      > log/compare_${wname}_${spec}.log 2>&1
    echo "DONE $wname pH=$pH target=$T -> $out"
  done
done
echo "ALL DONE"

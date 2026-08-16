#!/bin/bash
# E1b 验证扩展采样 — 4 PDB × (基线 + 3pH×3target) × 2 模型
# 设计见 session/2026-08-16_e1_validation_design.md
# 输出: code/output/e1_ext/{pdb}_{model}/...  日志: code/log/e1ext_*.log
source /home/baokun_yu/miniconda3/etc/profile.d/conda.sh
conda activate confumpnn
cd /data/nfs/IC/baokun_yu/ConfuMPNN/code
mkdir -p log

LIG=../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt
MOM=../MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt

for pdb in 1BC8 1CRN 1UBQ 2LZM; do
  PDBFILE=input/${pdb}.pdb
  # 提取 native 净电荷(pH7.4) 作为基线 target（probe 跑 1 条，仅用其 native 统计行）
  NAT=$(python run_guided.py --pdb $PDBFILE --pH 7.4 --target_charge 0 --num_samples 1 --weights "$LIG" \
      --out_dir output/e1_ext/_probe_${pdb} 2>&1 \
      | grep -oP 'native\s*:\s*charge=\s*[+-]?[0-9.]+' | grep -oP '[+-]?[0-9.]+$')
  echo "[probe] $pdb native_charge(pH7.4) = $NAT"
  if [ -z "$NAT" ]; then echo "!! 提取失败，跳过 $pdb"; continue; fi

  for wname in ligand mompnn; do
    if [ "$wname" = "ligand" ]; then W=$LIG; else W=$MOM; fi
    # ① 基线组：pH 7.4, target=native 电荷（贴近原结构，测结构保持）
    python run_guided.py --pdb $PDBFILE --pH 7.4 --target_charge=$NAT --num_samples 5 \
        --seed 111 --weights "$W" --out_dir output/e1_ext/${pdb}_${wname}/baseline \
        > log/e1ext_${pdb}_${wname}_baseline.log 2>&1
    echo "  DONE $pdb/$wname baseline"
    # ② 条件组：pH {4,7.4,9} × target {-5,0,+5}
    for pH in 4.0 7.4 9.0; do
      for T in -5 0 5; do
        python run_guided.py --pdb $PDBFILE --pH $pH --target_charge=$T --num_samples 3 \
            --seed 111 --weights "$W" --out_dir output/e1_ext/${pdb}_${wname}/pH${pH}_t${T} \
            > log/e1ext_${pdb}_${wname}_${pH}_${T}.log 2>&1
      done
    done
    echo "  DONE $pdb/$wname 全部条件"
  done
done
echo "ALL SAMPLING DONE"

#!/bin/bash
# exp7b S1 — 启动 7 蛋白（prot 4 + lig 3）各自后台进程于 cuda:6
# 用法: bash code/tests/run_exp7b_all.sh
source /home/baokun_yu/miniconda3/etc/profile.d/conda.sh
conda activate confumpnn
ROOT=/data/nfs/IC/baokun_yu/ConfuMPNN
cd $ROOT
mkdir -p log
PY=/home/baokun_yu/miniconda3/envs/confumpnn/bin/python

for pdb in 1AZM 1AS2 1BJ4 3MXB_A; do
  nohup env PYTHONPATH=$ROOT/code $PY $ROOT/code/tests/exp7b_pHgrid_sampler.py \
    --mode prot --device cuda:6 --proteins $pdb > log/exp7b_prot_$pdb.log 2>&1 &
  echo "launched prot $pdb pid $!"
done
for pdb in 5O60_E 1CGE 2FEO; do
  nohup env PYTHONPATH=$ROOT/code $PY $ROOT/code/tests/exp7b_pHgrid_sampler.py \
    --mode lig --device cuda:6 --proteins $pdb > log/exp7b_lig_$pdb.log 2>&1 &
  echo "launched lig $pdb pid $!"
done
echo "all launched"

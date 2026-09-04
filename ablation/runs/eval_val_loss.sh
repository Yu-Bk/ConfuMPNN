#!/usr/bin/env bash
# Forward val-loss replay for every ablation run (final epoch only).
# Protein family: tag v12_2 (FULL recipe), val set = holdout 1176 + valsupp 23 = 1199.
# Ligand  family: tag v14_ligand (FULL recipe), val set = v14_valset_805.
# Same eval recipe across all runs -> losses are horizontally comparable.
set -u
CONDA_PY=/home/baokun_yu/miniconda3/envs/confumpnn/bin/python
ROOT=/data/nfs/IC/baokun_yu/ConfuMPNN
cd "$ROOT"
DEV=cuda:6

PROT_LABELS="data/cath/labels_holdout_train.npz"
PROT_DOMPDB="data/cath/S40/dompdb_pdb"
PROT_SUPP="data/cath/labels_v12_3_valsupp.npz"
PROT_SUPPD="data/cath/S40/dompdb_valsupp"
LIG_LABELS="data/ligand_train/labels_v14_valset_805.npz"
LIG_DOMPDB="data/ligand_train/v14_valset_pdb"

# protein family final epochs = 10
for TAG in run_FULL run_nov12comp run_notarget run_noph run_nokeep; do
  OD="$ROOT/ablation/runs/prot/$TAG"
  if [ ! -f "$OD/finetune_epoch010.pt" ]; then echo "skip prot/$TAG (no final ckpt)"; continue; fi
  echo "[$(date '+%F %T')] eval prot/$TAG"
  PYTHONPATH=code "$CONDA_PY" code/tests/val_loss_curve.py --tag v12_2 \
      --ckpt_dir "$OD" --epoch_list 10 \
      --labels "$PROT_LABELS" --dompdb "$PROT_DOMPDB" \
      --supp_labels "$PROT_SUPP" --supp_dompdb "$PROT_SUPPD" \
      --device "$DEV" --out "$OD/val_loss.json"
done

# ligand family final epochs = 16
for TAG in run_FULL run_nov12comp run_notarget run_noA1 run_noph run_nokeep; do
  OD="$ROOT/ablation/runs/lig/$TAG"
  if [ ! -f "$OD/finetune_epoch016.pt" ]; then echo "skip lig/$TAG (no final ckpt)"; continue; fi
  echo "[$(date '+%F %T')] eval lig/$TAG"
  PYTHONPATH=code "$CONDA_PY" code/tests/val_loss_curve.py --tag v14_ligand \
      --ckpt_dir "$OD" --epoch_list 16 \
      --labels "$LIG_LABELS" --dompdb "$LIG_DOMPDB" \
      --device "$DEV" --out "$OD/val_loss.json"
done
echo "[$(date '+%F %T')] ALL VAL-LOSS EVALS COMPLETE"

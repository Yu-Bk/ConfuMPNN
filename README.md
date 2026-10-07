# ConfuMPNN

ConfuMPNN is a research-code framework for condition-aware protein sequence design with pH and charge objectives. It includes guided-sampling and condition-encoder fine-tuning entry points.

This is a framework repository, not a full rerun package. It does not include the datasets, input structures, model weights, evaluation outputs, or generated results needed to reproduce a complete run, and it makes no claim of end-to-end result reproducibility.

## Repository contents

- `code/run_guided.py` runs guided sequence sampling.
- `code/train_finetune.py` fine-tunes the condition encoder.
- `code/src/` contains condition encoding, sampling, charge calculations, loss functions, pKa and SASA helpers, and structure-aware filtering.
- `code/configs/` contains condition defaults and filter presets.
- `scripts/check_environment.py` checks local dependencies and expected sibling files without changing them.

## External source, checkpoints, and data

From the repository root, place the upstream projects as sibling directories alongside `code/`:

```bash
git clone https://github.com/dauparas/LigandMPNN.git LigandMPNN
git clone https://github.com/Qivon7/MoMPNN.git MoMPNN
```

The scripts import `data_utils.py` and `model_utils.py` from `LigandMPNN/`. By default, inference and training also expect this MoMPNN checkpoint relative to the repository root:

```text
MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt
```

Inference accepts `--weights` to select another checkpoint. For example, the original LigandMPNN checkpoint path used by the script is `LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt`.

Training defaults expect `data/cath/labels.npz` and `data/cath/S40/dompdb`. These CATH inputs are not included. Provide the required data and checkpoint paths before training. The upstream projects may have additional setup requirements; consult their documentation and licenses as well.

## Environment setup

The package versions in `requirements.txt` record the audited core environment (Python 3.11.16). They are a conservative runtime record, not a complete lockfile for every upstream dependency.

```bash
python3.11 -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux or macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch is required but intentionally omitted from `requirements.txt`: select a build that matches your operating system and CPU or accelerator setup using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Install any additional dependencies required by the selected upstream code.

FreeSASA is optional for basic use. Install `requirements-sasa.txt` when using SASA-dependent training losses. `Bio.PDB` is provided by Biopython, which is listed in the core requirements. Training catches SASA calculation failures and sets `frac_sasa=None`; losses that depend on SASA may therefore be skipped if FreeSASA or structure parsing is unavailable.

Run the read-only environment check from the repository root with:

```bash
python scripts/check_environment.py
```

## Example commands

Run inference from the repository root and replace the PDB placeholder with an input structure:

```bash
python code/run_guided.py --pdb path/to/protein.pdb --pH 7.4 --num_samples 10 --out_dir output/guided
```

The inference default checkpoint is `MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt`; pass `--weights` to use another checkpoint.

Training requires the external checkpoint and CATH inputs described above. A representative command is:

```bash
python code/train_finetune.py \
  --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt \
  --labels data/cath/labels.npz \
  --dompdb data/cath/S40/dompdb \
  --cfg code/configs/condition_defaults.yaml \
  --device cuda:0 \
  --epochs 30 \
  --max_domains 0 \
  --out_dir output/finetune \
  --log_progress output/finetune/train_progress.json \
  --log_file output/finetune/train.log
```

Choose a `--device` supported by your PyTorch installation; use `cpu` if appropriate for your setup. These commands document the CLI and expected paths; they do not establish that a particular environment or run has been validated. GPU execution is not claimed as tested here.

## Outputs and scope

Inference writes `seqs.fa` and `summary.json` under `--out_dir`. The FASTA contains generated candidates and the native sequence; the JSON records run settings and per-sequence charge and pI summaries. The default directory is `code/output/guided_<pdb>_pH<pH>`.

Training writes `finetune_epochNNN.pt` checkpoints and `condition_encoder_last.pt` under `--out_dir`. Progress and training logs are written to `--log_progress` and `--log_file`; the defaults are `code/log/train_progress.json` and `code/log/train.log`. These generated files are not evaluation results. This repository does not bundle benchmark outputs or evidence of a complete rerun.

## License and citation

This repository is licensed under the BSD 3-Clause License; see `LICENSE`. External projects and their files remain subject to their respective licenses. `CITATION.cff` identifies this software repository without inventing publication details. Please cite the associated paper once it is published.

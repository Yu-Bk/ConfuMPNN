# ConfuMPNN

ConfuMPNN is a research-code framework for condition-aware protein sequence design with pH and charge objectives. It includes guided-sampling and condition-encoder fine-tuning entry points.

## Repository contents

- `code/run_guided.py` runs guided sequence sampling.
- `code/train_finetune.py` fine-tunes the condition encoder.
- `code/src/` contains condition encoding, sampling, charge calculations, loss functions, pKa and SASA helpers, and structure-aware filtering.
- `code/configs/` contains condition defaults and filter presets.
- `scripts/check_environment.py` checks local dependencies and expected sibling files; an optional SASA mode performs a real structure preflight.
- `scripts/preflight_sasa.py` checks per-residue SASA coverage for one PDB.
- `scripts/fit_charge_calibration.py` fits a calibration JSON table from user-generated charge observations.

## External backbones, checkpoints, and data

From the repository root, place the upstream projects as sibling directories alongside `code/`:

```bash
git clone https://github.com/dauparas/LigandMPNN.git LigandMPNN
git clone https://github.com/Qivon7/MoMPNN.git MoMPNN
```

The scripts import `data_utils.py` and `model_utils.py` from `LigandMPNN/`. The [upstream LigandMPNN source](https://github.com/dauparas/LigandMPNN) and base checkpoints are external. Its [official README](https://github.com/dauparas/LigandMPNN#ligandmpnn) instructs running `bash get_model_params.sh ./model_params` from the LigandMPNN checkout. The code's LigandMPNN fallback checkpoint is `LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt`.

The [upstream MoMPNN repository](https://github.com/Qivon7/MoMPNN) includes its paper checkpoints. Inference and training default to this checkpoint relative to the repository root:

```text
MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt
```

Inference accepts `--weights` to select another backbone checkpoint and `--cond_encoder` to load a separate ConditionEncoder.

### ConfuMPNN ConditionEncoder weights

The [ConditionEncoder release](https://github.com/Yu-Bk/ConfuMPNN/releases/tag/conditionencoder) provides this separate weight layer; it does not provide the LigandMPNN or MoMPNN backbone weights. Download the encoder files into `weights_release/` with GitHub CLI:

```bash
gh release download conditionencoder --repo Yu-Bk/ConfuMPNN --pattern "condition_encoder*.pt" --dir weights_release
```

| Checkpoint | Mode |
| --- | --- |
| `condition_encoder_v12_2_last.pt` | v12.2, short-chain protein |
| `condition_encoder_v12_3_last.pt` | v12.3, long-chain protein |
| `condition_encoder_v14_ligand_epoch050.pt` | v14, ligand mode |

Pass the selected file with `--cond_encoder` during inference.

Training defaults expect `data/labels.npz` and `data/S40/dompdb`. These CATH inputs are not included. Provide the required data and checkpoint paths before training. The upstream projects may have additional setup requirements; consult their documentation and licenses as well.

## Environment setup

The package versions in `requirements.txt` record the audited core runtime (Python 3.11.16), not a complete lockfile for upstream projects.

```bash
python3.11 -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux or macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch is required but omitted from `requirements.txt` because its build must match your operating system, accelerator, and CUDA setup. Select a compatible build with the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), and install any additional dependencies required by the selected upstream code.

FreeSASA is optional for basic use. Install the existing `requirements-sasa.txt` when using SASA-dependent training losses. `Bio.PDB` is provided by Biopython, which is listed in the core requirements. Training catches SASA calculation failures and sets `frac_sasa=None`; losses that depend on SASA may therefore be skipped if FreeSASA or structure parsing is unavailable.

Check one real PDB before training with SASA-dependent losses:

```bash
python scripts/preflight_sasa.py --pdb path/to/single_chain.pdb --threshold 0.25 --json-out sasa_preflight.json
```

The environment checker can run the same preflight:

```bash
python scripts/check_environment.py --sasa-pdb path/to/single_chain.pdb --sasa-threshold 0.25 --sasa-json sasa_preflight.json
```

The preflight requires one PDB model and exactly one protein chain containing standard amino acids with Cα atoms, unique residue identifiers, and no insertion codes. Other chains and residues, such as RNA, ligands, or water, may remain as structural context. It checks raw FreeSASA areas against the public helper's residue mapping and reports surface/core counts; neither region must be nonempty. It checks one structure's SASA mapping only. It does not verify training-data alignment, pocket masks, loss activation, or GPU execution. An import-only environment check does not establish that every training sample's SASA losses will be active.

`core_charge` in the online structure filter is a geometric burial proxy based on Cα-neighbor counts within 10 Å; it does not use FreeSASA. Training-time fractional-SASA masks are a separate calculation.

## Sampling routes and calibration

| Route | Command options | Behavior |
| --- | --- | --- |
| ConditionEncoder | `--cond_encoder PATH` | Injects the pH/charge condition without ChargeLookahead or StructureAwareFilter. `--calibrate auto` reads a precomputed table, preferring a per-protein entry and then a global fit. |
| ChargeLookahead + StructureAwareFilter | Omit `--cond_encoder`; optionally set `--preset` and `--strength` | Uses the guided-sampling route and its dynamic charge bias plus structural logit filter. The default `same_sign_cluster` rule uses an 8 Å radius and threshold 4. Current rule version: `local_density_v2_2026-09-16`, which counts decoded same-sign residues per valid target-position neighborhood. |
| Hybrid | Add `--hybrid --cond_encoder PATH` | Combines ConditionEncoder injection with ChargeLookahead and StructureAwareFilter. Requires `--cond_mode conditioned`; `baseline` does not inject the encoder condition and cannot be combined with `--hybrid`. ChargeLookahead uses the requested target; ConditionEncoder receives `target_eff` after any applied calibration. |

This release replaces the older 8 Å connected-component rule with per-position local-density counts. The filter's trigger locations and generated sequences can therefore differ from those produced by older checkouts.

Example hybrid command from the repository root (replace the PDB and checkpoint paths):

```bash
python code/run_guided.py \
  --pdb path/to/protein.pdb \
  --pH 7.4 \
  --target_charge 0 \
  --weights path/to/backbone_checkpoint.pt \
  --cond_encoder path/to/condition_encoder.pt \
  --hybrid \
  --preset default \
  --strength 0.5
```

This command uses the requested target for ChargeLookahead. If a calibration table is provided, ConditionEncoder receives the adjusted `target_eff`.

### Charge calibration table

No calibration table is bundled at the optional default path `output/charge_calibration_v12_2.json`. With no table, the default YAML fallback is disabled, so `--calibrate auto` or `--calibrate global` reports that calibration was not applied and uses the requested target unchanged. `--calibrate global` requires a global table fit. `--calibrate off` and `--no_calibration` both disable calibration. For conditioned sampling with calibration active, an explicitly supplied missing or malformed table raises an error instead of silently using an uncalibrated target.

Calibration JSON uses a `global` fit and an optional `per_protein` mapping, each entry containing numeric `slope` and `intercept`; inference applies `target_eff = (target_charge - intercept) / slope`. The per-protein entries are precomputed lookups. They are not fitted on the current protein during inference and do not constitute an on-site calibration procedure. To build a table, first collect actual target/observed charge pairs from generated sequences under the same backbone and encoder checkpoints, pH, design/fixed-position mask, and sampling temperature. `observed_charge` is the charge computed from the generated sequence, not an expected charge from logits. The fitting tool does not generate observations or invent coefficients.

CSV input schema (header shown):

```text
protein,target_charge,observed_charge
```

Fit the table from that user-created CSV, then pass it explicitly:

```bash
python scripts/fit_charge_calibration.py \
  --input path/to/user_observations.csv \
  --output output/charge_calibration_user.json \
  --label "checkpoint/pH/mask/temperature cohort"

python code/run_guided.py --pdb path/to/protein.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder path/to/condition_encoder.pt \
  --calibration_file output/charge_calibration_user.json
```

The CSV requires finite numeric charges and at least two distinct targets for the global fit. Per-protein fits are emitted only when that protein has at least two distinct targets and a positive usable slope; otherwise that entry is skipped and the table's global fit remains available. Existing tables that contain no usable fit are rejected. A fitted table is only appropriate for the checkpoint, pH, mask, and temperature represented by its observations; a new setup needs its own observations and fit.

Run the read-only environment check from the repository root with:

```bash
python scripts/check_environment.py
```

## Optional downstream evaluation

These tools are optional and must be installed separately from the core runtime.

| Tool | Role | Installation boundary |
| --- | --- | --- |
| [ESMFold](https://github.com/facebookresearch/esm) | Predicts structures and pLDDT. | Use an isolated compatible environment for its separate PyTorch, OpenFold, and CUDA requirements. |
| [US-align](https://www.zhanggroup.org/US-align/) | External executable for structure alignment and TM-score. | Install separately; it is not a pip dependency. |
| [Protein-Sol](https://protein-sol.manchester.ac.uk/software) | Optional local sequence-solubility predictor. | Requires separate installation; upstream software is shell/Perl based. |
| [TemBERTure](https://github.com/ibmm-unibe-ch/TemBERTure) | External sequence-based melting-temperature predictor. | Has its own code and weights. |

This checkout does not bundle the validation code, workflows, or results for these optional checks. Installing the tools alone does not reproduce paper results; these are computational predictions, not wet-lab validation.

## Tests

These CPU tests cover the environment checker, SASA preflight, filter neighborhoods, hybrid callback forwarding, and calibration fitting/metadata. They do not run model training or GPU inference. Run them from the repository root with:

```bash
python -m unittest discover -s tests
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

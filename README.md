# ConfuMPNN

ConfuMPNN is a research code framework for exploring condition-aware protein sequence design with pH and charge objectives. It includes guided-sampling and condition-encoder fine-tuning entry points.

## Layout

- `code/run_guided.py` — guided sequence-sampling entry point.
- `code/train_finetune.py` — condition-encoder fine-tuning entry point.
- `code/src/` — condition encoding, sampling, charge calculations, loss functions, pKa and SASA helpers, and structure-aware filtering.
- `code/configs/` — condition defaults and filter presets.

## External dependencies

Base model implementations and checkpoints are not included. Obtain them separately from the official [LigandMPNN](https://github.com/dauparas/LigandMPNN) and [MoMPNN](https://github.com/Qivon7/MoMPNN) repositories. Install the Python dependencies required by those projects and by the selected scripts, and configure the model and input paths for your environment.

## Limitations

This repository contains research code. Required structures, datasets, model weights, and evaluation outputs are not bundled. Reproducing a particular result depends on the external inputs, checkpoints, software environment, and workflow used; this repository does not promise complete reproduction of any dataset or result.

## Citation

Citation information for the ConfuMPNN work will be added when a publication becomes available.

# DLAIO-CDR

Code for two-stage deep learning correction of dopamine transporter SPECT, accompanying *A Deep Learning-based All-in-one Correction Pipeline for Dopamine Transporter SPECT Using Realistic Simulations of a Digital Brain Phantom Population*.

This repository provides the DLAIO-CDR network architecture, preprocessing and inference code, a paired-volume dataset helper, a phantom evaluation wrapper, a synthetic example, and a five-fold checkpoint manifest.

## Scope

The inference entry point supports DLAIO-CDR only. Numerical verification covers one synthetic example with the fold 1 model pair on CPU. Both stage outputs were identical to their archived counterparts. This does not establish reproduction of all historical training runs, execution of every fold, CUDA numerical agreement, or installation on every platform. See [verification/summary.json](verification/summary.json), [TRAINING.md](TRAINING.md), [EVALUATION.md](EVALUATION.md), and [release_notes.md](release_notes.md).

## Environment

The verified Windows environment used Python 3.9.17, PyTorch 2.0.1 (CUDA 11.7 build), NumPy 1.25.2, and SimpleITK 2.2.1. The recorded package versions are provided in `requirements_recorded.txt`; they are not a cross-platform installation lockfile.

## Model checkpoints

Pretrained model weights are available from the [v1.0.0 release](https://github.com/weberhuang312-alt/DLAIO-CDR/releases/tag/v1.0.0). Each `DLAIO-CDR-foldN-weights.zip` contains the primary and secondary model for one fold. Download the fold(s) you need and extract the archives into a common directory. `SHA256SUMS.txt` in the release provides checksums for the ZIP files.

`model_manifest.csv` identifies all ten checkpoints (two stages for each of five folds), including SHA256 hashes and pairings. The inference entry point checks both checkpoint hashes before running. Set `WEIGHTS_ROOT` to the directory containing the extracted `08_selected_models` directory.

Arrange the selected checkpoints under `WEIGHTS_ROOT` using the filenames below. Renaming a checkpoint file does not change its SHA256; the manifest verifies the file contents before inference.

```text
08_selected_models/DLAIO-CDR/fold1/stage1/G_AB.pth
08_selected_models/DLAIO-CDR/fold1/stage2/G_AB.pth
# Equivalent stage1/stage2 directories for folds 2-5
```

## Inference

From the repository root, replace `WEIGHTS_ROOT` with the directory containing the checkpoint layout above. Choose a new output directory:

```bash
python code/infer.py --config configs/cdr.json --input examples/example_simulated_nc.nii --output outputs/example_cpu --weights-root WEIGHTS_ROOT --strategy CDR --fold 1 --device cpu
```

Use `--device cuda` for a compatible CUDA environment. Unavailable CUDA raises an error rather than silently changing devices. `--fold` selects a model pair from the manifest, and both checkpoint hashes are checked before inference.

The output directory contains `stage1.nii.gz`, `stage2.nii.gz`, and `run_record.json`.

## Input and preprocessing

- Input: a finite 128 x 128 x 128 reconstructed NC NIfTI volume on the experiment's stored intensity scale.
- SimpleITK supplies Z-Y-X array indexing. Batch and channel axes produce N-C-Z-Y-X tensors.
- Each network input passes through float16 and then float32, including the first-stage prediction before the second network. No min-max or z-score intensity normalization is applied.
- This entry point adds no cropping, padding, flipping, spatial resampling, filtering, output rescaling, clipping, or activity calibration. Other input sizes are rejected.
- Output spatial metadata are copied from the input. This does not establish the true physical sampling of historical data, and does not reproduce historical exporters that only reset spacing in file headers.

The native 16-channel DLAIO-sim implementation and other correction strategies are outside this entry point's verified scope.

## Files

| File | Purpose |
|---|---|
| `code/infer.py` | Two-stage inference with explicit paths, fold selection, and SHA256 checks |
| `code/architecture32.py` | Matching 32-channel architecture snapshot |
| `code/preprocessing.py` | Shared array and precision preparation |
| `code/paired_dataset.py` | Paired CSV dataset helper; not a recovered historical training pipeline |
| `code/evaluation/` | Existing custom NMSE function and a MATLAB phantom evaluation wrapper |
| `examples/` | Synthetic example input |
| `model_manifest.csv` | Checkpoint identities and pairings |
| `verification/` | Recorded single-example verification results and environment |

No patient images or patient lists are included. No new open-source license is assigned by this release; existing third-party rights remain applicable.

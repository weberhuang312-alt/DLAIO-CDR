# Implementation scope and limitations

- Verification covers one synthetic historical example, DLAIO-CDR fold 1, on CPU with five PyTorch threads. Both outputs were byte-identical to the archived example. All ten checkpoint identities were statically checked; the other folds and CUDA agreement were not executed in this check.
- Pretrained DLAIO-CDR checkpoints are distributed separately through the [v1.0.0 release](https://github.com/weberhuang312-alt/DLAIO-CDR/releases/tag/v1.0.0), with one archive per fold and two stages per archive. All ten checkpoint hashes match `model_manifest.csv`; `SHA256SUMS.txt` verifies the downloadable archives. The checkpoint contents are unchanged.
- The full historical training-loader associations, original validation-based checkpoint selection, parts of upstream resampling, and physical image sampling remain incompletely documented. The portable entry point is not a complete reconstruction of every historical training or export procedure.
- Changing NIfTI spacing metadata does not resample an array. The separate historical division by 19.683 is not applied by this inference entry point.
- The Gaussian postfilter used for a conventional comparator, upstream image processing, and evaluation cropping are separate from the network preparation implemented here.
- The recorded installed package versions are not a cross-platform-tested lockfile. No new open-source license is assigned; original third-party rights are unchanged.

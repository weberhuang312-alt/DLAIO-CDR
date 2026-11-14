# Evaluation scope

`code/evaluation/cal_Weber_NMSE.m` is the original custom NMSE implementation. `evaluate_pair.m` provides a relocatable single-pair wrapper. MATLAB R2023a and Image Processing Toolbox are required for the default three-dimensional `ssim` call; toolbox source is not distributed.

The wrapper takes input and reference arrays with common XYZ indexing and uses the phantom crop `x=20:100, y=20:100, z=1:60` (inclusive), without an additional brain mask. The custom NMSE normalization is defined by the supplied function and is not squared error divided by reference energy. Clinical map-based evaluation uses a separate protocol and is not reproduced by this wrapper.

The wrapper was syntax/source checked but was not executed in MATLAB during this code handoff. The numerical check in `verification/summary.json` compares the two inference outputs with their archived example outputs; it is not a new cohort-level metric experiment.

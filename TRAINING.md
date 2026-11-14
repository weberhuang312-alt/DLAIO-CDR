# Training scope

The repository contains the architecture and a relocatable dataset helper, not a complete, verified historical training command. Inference reproducibility and historical training reproducibility are distinct.

`code/paired_dataset.py` reads an explicit CSV with `role`, `input`, and `target` columns. Both volumes use 128-cubed arrays, float16-to-float32 preparation, and identity intensity handling, without augmentation or spatial resampling. This helper documents the confirmed revision data contract and is not an authentic recovered historical loader.

Clinical stage 1 uses NC-to-CDR pairs. Phantom stage 2 uses predicted intermediate simulation images and their known activity references. Historical loader associations and the original checkpoint-selection criteria are not fully established. The manifest identifies checkpoints selected for the delivered revision evaluation; it does not label them as validation-optimal.

No training is executed by the inference entry point. No private patient manifest or laboratory training source is distributed here.

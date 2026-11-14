"""Array-space preparation, shared by both stages."""
import numpy as np
import torch
def prepare(array):
    # SimpleITK yields ZYX. Preserve indexing; quantization is NOT normalization.
    a=np.asarray(array).astype('float16').astype('float32')
    if a.shape!=(128,128,128) or not np.isfinite(a).all():raise ValueError('Invalid input afterfloat16 roundtrip')
    # N,C,Z,Y,X; no transpose, cropping, padding, flips, filtering or scaling.
    return torch.from_numpy(a)[None,None]

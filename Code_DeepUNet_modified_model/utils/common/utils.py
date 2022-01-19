"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from math import sqrt
import h5py
import numpy as np

def save_reconstructions(reconstructions, out_dir, targets=None, inputs=None):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
        target (np.array): target array
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)
            if targets is not None:
                f.create_dataset('target', data=targets[fname])
            if inputs is not None:
                f.create_dataset('input', data=inputs[fname])


def ssim_loss(gt, pred, maxval=None):
    """Compute Structural Similarity Index Metric (SSIM)
       ssim_loss is defined as (1 - ssim)
    """
    maxval = gt.max() if maxval is None else maxval

    ssim = 0
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    ssim = ssim / gt.shape[0]
    return 1 - ssim


def nmse(gt, pred):
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred, maxval = None):
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def fftc(data, axes=(-2, -1), norm="ortho"):
    """
    Centered fast fourier transform
    """
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(data, axes=axes), 
                    axes=axes, 
                    norm=norm), 
        axes=axes
    )


def ifftc(data, axes=(-2, -1), norm="ortho"):
    """
    Centered inverse fast fourier transform
    """
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(data, axes=axes), 
                     axes=axes, 
                     norm=norm), 
        axes=axes
    )

def rss_combine(data, axis, keepdims=False):
    return np.sqrt(np.sum(np.square(np.abs(data)), axis, keepdims=keepdims))


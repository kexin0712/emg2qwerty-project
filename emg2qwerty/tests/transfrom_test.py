# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from emg2qwerty.transforms import Downsample, MagnitudeWarping, WaveletDecomposition

def test_magnitude_warping_noop_when_sigma_zero():
    x = torch.randn(20, 2, 16)
    y = MagnitudeWarping(sigma=0.0)(x)
    assert torch.equal(y, x)


def test_magnitude_warping_preserves_shape():
    x = torch.randn(20, 2, 16)
    y = MagnitudeWarping(sigma=0.2, knot=4, independent_channels=True)(x)
    assert y.shape == x.shape
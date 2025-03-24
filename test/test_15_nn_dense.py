import numpy as np
import pytest
import torch

from qcd_ml.base.operations import link_gauge_transform, v_gauge_transform
from qcd_ml.nn.dense import v_Dense
from qcd_ml.qcd.static import gamma


def test_v_Dense_reverse(psi_test):
    layer = v_Dense(2, 2)
    layer.weights.data[:,:] = gamma[1] / np.sqrt(2)
    layer.weights.data[1,1] = -gamma[1] / np.sqrt(2)

    features_in = torch.stack([psi_test, psi_test])
    features_in = torch.randn_like(features_in)

    features_out = layer.forward(features_in)
    features_reverse = layer.reverse(features_out)

    assert torch.allclose(features_in, features_reverse)

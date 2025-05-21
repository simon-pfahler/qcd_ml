import torch

from qcd_ml.nn.non_gauge import C_Convolution


def test_C_Convolution_circular():
    n_input = 2
    n_output = 1
    stride = 3

    layer = C_Convolution(n_input, n_output, stride)
    input_features = torch.randn(n_input, 8,8,8,16, 3,3, dtype=torch.cdouble)

    features_out = layer.forward(input_features)

    shifts = torch.randint(1, 8, [3]).tolist() + torch.randint(1, 16, [1]).tolist()
    dims = list(range(1, 5))
    input_features_c = torch.roll(input_features, shifts, dims)

    features_out_c = layer.forward(input_features_c)
    assert torch.allclose(features_out, torch.roll(features_out_c, [-s for s in shifts], dims))


def test_C_Convolution_3d():
    n_input = 2
    n_output = 1
    stride = 3

    layer = C_Convolution(n_input, n_output, stride, nd=3)
    input_features = torch.randn(n_input, 8,8,16, dtype=torch.cdouble)

    features_out = layer.forward(input_features)

    pad3d = torch.nn.CircularPad3d(layer.padding)
    conv3d = torch.nn.Conv3d(n_input, n_output, layer.kernel_size, layer.stride)
    conv3d.weight = torch.nn.Parameter(layer.weights.data.transpose(0, 1))
    conv3d.bias = layer.biases

    features_out_torch = conv3d.forward(pad3d(input_features))
    assert torch.allclose(features_out, features_out_torch)

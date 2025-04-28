"""
------------

"""
import torch


class LGE_Exp(torch.nn.Module):
    r"""
    Provides an exponentiation layer for matrix-like fields acting on
    gauge links, i.e.,

    .. math::
        U_\mu(x) \rightarrow \exp\left(\sum\limits_j \beta_{\mu,i} W_{i}(x)\right) U_\mu(x)
    """

    def __init__(self, n_features_in, matrix_mode='Ah'):
        """

        :param n_features_in:
        :param matrix_mode: Type of anti-hermitian matrix used in the exponent
        Ah -> exp(anti-hermitian)
        H -> exp(i * hermitian)
        Ah_H -> exp(anti-hermitian + i * hermitian)
        """
        super(LGE_Exp, self).__init__()
        self.matrix_mode = matrix_mode
        if matrix_mode == "Ah" or matrix_mode == "Ah_H":
            self.ah_weights = torch.nn.Parameter(torch.randn(4, n_features_in, dtype=torch.double))
        if matrix_mode == "H" or matrix_mode == "Ah_H":
            self.h_weights = torch.nn.Parameter(torch.randn(4, n_features_in, dtype=torch.double))
        if matrix_mode not in ["Ah", "H", "Ah_H"]:
            raise Exception("invalid matrix_mode. The possibilities are:\n"
                            "Ah -> exp(anti-hermitian)\n"
                            "H -> exp(i * hermitian)\n"
                            "Ah_H -> exp(anti-hermitian + i * hermitian)")

    def forward(self, U, W):
        transform_matrix = torch.zeros(U.shape, dtype=torch.cdouble)
        W_adj = W.adjoint()

        if self.matrix_mode == "Ah" or self.matrix_mode == "Ah_H":
            anti_hermitian = W - W_adj
            traceless_anti_hermitian = anti_hermitian - torch.einsum("itxyzaa,bc->itxyzbc", anti_hermitian, torch.eye(3)) / 3

            transform_matrix += torch.matrix_exp(torch.einsum("ui,i...->u...",
                                                              1j * self.ah_weights, -1j * traceless_anti_hermitian))

        if self.matrix_mode == "H" or self.matrix_mode == "Ah_H":
            hermitian = W + W_adj
            traceless_hermitian = hermitian - torch.einsum("itxyzaa,bc->itxyzbc", hermitian, torch.eye(3)) / 3

            transform_matrix += torch.matrix_exp(torch.einsum("ui,i...->u...",
                                                              1j * self.h_weights, -1j * traceless_hermitian))

        return torch.einsum("uabcdij,uabcdjk->uabcdik", transform_matrix, U)

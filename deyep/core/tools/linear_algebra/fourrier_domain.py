# Global import
import numpy as np
# Local import


def get_fourrier_basis_from_series(ax_s, ax_basis, return_coef=False):

    ax_res = np.round(np.real(ax_s.dot(ax_basis.transpose().conjugate())))
    if return_coef:
        return [i for i in ax_res.nonzero()[0]], [ax_res[i] for i in ax_res.nonzero()[0]]
    else:
        return [i for i in ax_res.nonzero()[0]]


def get_fourrier_series(N, k):

    coef = np.exp(np.complex(np.log(1. / np.sqrt(N)), - (2. * np.pi * k) / N))
    return coef * np.exp(-1j * (2. * np.pi * k * np.arange(-1, N - 1)) / N)

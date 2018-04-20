# Global import
import numpy as np


def get_fourrier_coef(N, k):
    return np.exp(np.complex(np.log(1. / np.sqrt(N)), (2. * np.pi * k) / N))


def get_fourrier_key(coef):

    # Get N
    N = int(pow(np.linalg.norm(coef), 2))

    # Get k
    k = int((np.imag(coef) * N) / (2. * np.pi))

    return N, k


def get_fourrier_series(coef):
    # Get N
    N = int(pow(np.linalg.norm(coef), 2))

    return np.array([pow(coef, t) for t in range(N)])


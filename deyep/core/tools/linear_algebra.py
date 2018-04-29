# Global import
import numpy as np


def get_fourrier_coef(N, k):
    return np.exp(np.complex(np.log(1. / np.sqrt(N)), - (2. * np.pi * k) / N))


def get_fourrier_key(coef):

    # Get N
    N = int(np.round(1. / pow(np.linalg.norm(coef), 2)))

    # Get k
    k = (np.angle(coef) * N) / (2. * np.pi)

    if k <= 0:
        k = int(np.round(- k))
    else:
        k = int(np.round(N - k))

    return N, k


def get_fourrier_series(coef):

    # Get N
    N, k = get_fourrier_key(coef)

    return np.array([coef * np.exp(np.complex(0, - (2. * np.pi * k * t) / N)) for t in range(-1, N - 1)])


def inner_product(x, y):
    return x.dot(y.conjugate())



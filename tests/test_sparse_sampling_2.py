import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# export PYTHONPATH="/home/erepie/deyep/"


def gradient(y):
    red = [(0.0, 0.0, 0.0), (0.5, y, y), (1.0, 0.0, 0.0)]
    green = [(0.0, 0.0, 0.0), (0.5, y, y), (1.0, y, y)]
    blue = [(0.0, y, y), (0.5, y, y),(1.0,0.0,0.0)]
    colordict = dict(red=red, green=green, blue=blue)
    bluegreenmap = LinearSegmentedColormap('bluegreen', colordict, 256)
    return bluegreenmap


def compute_norm(l0, size_node_, ax_bits_, ax_factors_, K_):
    res = 0
    for i in np.arange(l0, size_node_ + 1):
        res += compute_norm_level(i, size_node_, ax_bits_, ax_factors_, K_)

    return res


def compute_norm_level(i, size_node_, ax_bits_, ax_factors_, K_):
    ax_bits_ = np.array([f_binom(i, size_node_, ax_bits_[k]) for k in range(K_)])
    res = ax_factors_.dot(ax_bits_)
    return res


def compute_product(la_0, size_node_a, lb_0, size_node_b, n, ax_bits_, ax_factors_, factor=1.):

    res = 0
    for i in np.arange(max(la_0, n), size_node_a + 1):
        for j in np.arange(lb_0, size_node_b + 1):
            res += compute_product_level(n, i, size_node_a, j, size_node_b, ax_bits_, ax_factors_, factor=factor)

    return res


def compute_product_level(n, i, size_node_a, j, size_node_b, ax_bits_, ax_factors_, factor=1.):

    ax_bits_ = np.array(
        [f_binom(i - n, size_node_a - n, ax_bits_[k]) * f_binom(j, size_node_b, ax_bits_[k]) for k in range(K)]
    )
    res = ax_factors_.dot(ax_bits_ * factor)
    return res


def compute_theoretical_max(size_node, K, pf, pg, k=None, round='round'):
    if k is None:
        k_star = compute_theoretical_max_factors(K, pf, round=round)
    else:
        k_star = k
    pg_star = 1 - binom.pmf(0, k_star, pg)
    return size_node * pg_star


def compute_theoretical_max_bits(size_node, pg_star):
    return size_node * pg_star


def compute_theoretical_max_factors(K, pf, round='round'):
    if round == 'round':
        return np.round(K * pf)
    elif round == 'floor':
        return np.floor(K * pf)
    elif round == 'ceil':
        return np.ceil(K * pf)

###### GLOBAL PARAMS
K, n = 60, 1000
pf, pg = 0.1, 50. / n
f_binom = binom.pmf

ax_factors = np.array([f_binom(k - 1, K - 1, pf) for k in range(1, K + 1)])
ax_coef = np.array([1. / (1. - f_binom(0, k, pg)) for k in range(1, K + 1)])
ax_bits = np.array([f_binom(i, K, pg) for i in range(1, K + 1)])

# Line 1
ax_l1 = ax_factors.dot(ax_coef) * ax_bits

ax_l2 = np.zeros(K)
for i in range(1, K + 1):

    ax_factors_ = (float(K - i) * f_binom(i, i, 1 - pf)/ K) * np.array([f_binom(k - 1, K - i - 1, pf) for k in range(1, K - i + 1)])
    ax_coef_ = np.array([1. / (1. - f_binom(0, k, pg)) for k in range(1, K - i + 1)])
    ax_bits_ = np.array([f_binom(i, K, pg) for k in range(1, K - i + 1)])

    # Fill line 2
    ax_l2[i-1] = (ax_factors_ * ax_coef_).dot(ax_bits_)

import IPython
IPython.embed()






ax_factors = (1. / (1. - f_binom(0, K, pf))) * np.array([f_binom(k, K, pf) for k in range(1, K + 1)])
ax_coef = np.array([1. / (1. - f_binom(0, k, pg)) for k in range(1, K + 1)])
ax_bits = np.array([f_binom(i, K, pg) for i in range(1, K + 1)])

# Line 1
ax_l1 = ax_factors.dot(ax_coef) * ax_bits

ax_l2 = np.zeros(K)
for i in range(1, K + 1):

    ax_factors_ = (1. / (1. - f_binom(0, K, pf))) * np.array([f_binom(k, K, pf) for k in range(1, K - i + 1)])
    ax_coef_ = np.array([1. / (1. - f_binom(0, k, pg)) for k in range(1, K - i + 1)])
    ax_bits_ = np.array([f_binom(i, K - k, pg) * f_binom(k, k, 1 - pg) for k in range(1, K - i + 1)])

    # Fill line 2
    ax_l2[i-1] = (ax_factors_ * ax_coef_).dot(ax_bits_)


# line wothout any work on equation
ax_l3 = np.zeros(K)
for i in range(1, K + 1):
    _ax_factors = (1. / (1. - f_binom(0, K, pf))) * np.array([f_binom(k, K, pf) for k in range(1, K + 1)])
    _ax_bits = np.zeros(K)
    for k in range(1, K + 1):
        lbound, rbound = max(i + k - K, 1), min(i, k)
        ax_g1 = (1. / (1. - f_binom(0, k, pg))) * np.array([f_binom(u, k, pg) for u in range(lbound, rbound + 1)])
        ax_g2 = np.array([f_binom(i - u, K - k, pg) for u in range(lbound, rbound + 1)])
        _ax_bits[k - 1] = ax_g1.dot(ax_g2)

    ax_l3[i - 1] = _ax_factors.dot(_ax_bits)



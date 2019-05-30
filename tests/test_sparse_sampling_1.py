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
pf, pg = 0.01, 50. / n
f_binom = binom.pmf
ax_factors = np.array([f_binom(k, K, pf) for k in range(1, K + 1)])
ax_bits = np.array([1 - f_binom(0, k, pg) for k in range(1, K + 1)])

level_offset_a, size_node_a = 1, 100
level_offset_b, size_node_b = 1, 100

# APPLICATION 1: PDF decomposition of norm of a random node
print '---- Computing APP 1 ----'
size_node = 100
theoretical_max_a = compute_theoretical_max(size_node, K, pf, pg, round='floor')
theoretical_max_b = compute_theoretical_max(size_node, K, pf, pg, round='ceil')

d_pdf_node, d_pdf_sources, ax_pdf = {}, {}, np.array([0.] * size_node)
# Get decomposition
for k in range(K):
    d_pdf_sources[k] = {'x': size_node * (1 - binom.pmf(0, k + 1, pg)), 'y': ax_factors[k]}
    ax_pdf_node = np.array([0.] * size_node)
    for l0 in range(1, size_node + 1):
        ax_pdf_node[l0 - 1] = f_binom(l0, size_node, ax_bits[k])

    d_pdf_node[k] = ax_pdf_node.copy()

# Get PDF
for l0 in range(1, size_node + 1):
    ax_pdf[l0 - 1] = compute_norm_level(l0, size_node, ax_bits, ax_factors, K)

fig_decomposition_pdf = plt.figure()
plt.subplot(211)
for k, ax_pdf_ in d_pdf_node.items():
    plt.plot(np.arange(1, size_node + 1), ax_pdf_, alpha=1. - ((k * 4.) / 100))

plt.scatter(
    [d_pdf_sources[k]['x'] for k in d_pdf_sources.keys()],
    [d_pdf_sources[k]['y'] for k in d_pdf_sources.keys()],
    s=10, c='green'
)
plt.title('Decomposition of the PDF of norm of a random node')

plt.subplot(212)
plt.plot(np.arange(1, size_node_a + 1), ax_pdf)
plt.axvline(x=theoretical_max_a, color='b')
plt.axvline(x=theoretical_max_b, color='r')
plt.title('PDF of norm of a random node')


# APPLICATION 2: last level precision against size of node with different percent of overlap
# SPECIFIC PARAMS
print '---- Computing APP 2 ----'
ax_factors = np.array([f_binom(k, K - 1, pf) for k in range(1, K)])
ax_bits = np.array([1 - f_binom(0, k, pg) for k in range(1, K)])
case_i = range(1, 20)
min_size, max_size, step = 20, 250, 5
d_plot = {}

for i in case_i:
    d_plot.update({i: np.array([0.] * len(range(min_size, max_size, step)))})

    for k, size_node in enumerate(range(min_size, max_size, step)):
        size_node_ = size_node - i

        norm_a = compute_norm(size_node_, size_node_, ax_bits, ax_factors, K - 1)
        norm_b = compute_norm(size_node, size_node, ax_bits, ax_factors, K - 1)
        d_plot[i][k] = (pf * norm_a) / ((pf * norm_a) + ((1 - pf) * norm_b))

fig_precision_1 = plt.figure()
for k, (i, ax_v) in enumerate(d_plot.items()):
    plt.plot(np.array(range(min_size, max_size, step)), ax_v, alpha=1. - ((k * 15.) / 100))

plt.title("Precision of node's last level against size of node with different percent of overlap")

# APPLICATION 3: Last level precision against percent of overlap with different size of node
# SPECIFIC PARAMS
print '---- Computing APP 3 ----'
ax_factors = np.array([f_binom(k, K - 1, pf) for k in range(1, K)])
ax_bits = np.array([1 - f_binom(0, k, pg) for k in range(1, K)])
case_i, node_size = 20, 100
ax_precision = np.array([0.] * case_i)

for k, i in enumerate(np.arange(case_i, dtype=float) / (2 * case_i)):
    size_node_ = size_node - int(i * size_node)
    norm_a = compute_norm(size_node_, size_node_, ax_bits, ax_factors, K - 1)
    norm_b = compute_norm(size_node, size_node, ax_bits, ax_factors, K - 1)
    ax_precision[k] = (pf * norm_a) / ((pf * norm_a) + ((1 - pf) * norm_b))

fig_precision_2 = plt.figure()
plt.plot(np.array(np.arange(case_i, dtype=float) / (2 * case_i)), ax_precision)

plt.title("Precision of node's last level against percent of overlap with different size of node")


# APPLICATION 4: Precision against level with different percent of overlap
# SPECIFIC PARAMS
print '---- Computing APP 4 ----'
ax_factors = np.array([f_binom(k, K - 1, pf) for k in range(1, K)])
ax_bits = np.array([1 - f_binom(0, k, pg) for k in range(1, K)])
size_node = 100
fig_precision_3 = plt.figure()

for i in [10, 30, 50]:
    ax_precision = np.array([0.] * size_node)
    size_node_ = size_node - i
    for l0 in range(i, size_node + 1):
        l0_ = l0 - i
        norm_a = compute_norm(l0_, size_node_, ax_bits, ax_factors, K - 1)
        norm_b = compute_norm(l0, size_node, ax_bits, ax_factors, K - 1)
        ax_precision[l0 - 1] = (pf * norm_a) / ((pf * norm_a) + ((1 - pf) * norm_b))

    plt.plot(np.array(range(1, size_node + 1)), ax_precision, 'k', alpha=1. - (float(i) / 75.))

plt.title("Precision of node's levels with different overlap value")
plt.show()


# APPLICATION 5: Precision against level with different percent of overlap
# SPECIFIC PARAMS
print '---- Computing APP 1 ----'
ax_factors = np.array([f_binom(k, K - 1, pf) for k in range(1, K)])
ax_bits = np.array([1 - f_binom(0, k, pg) for k in range(1, K)])
size_node, i, size_node_ = 100, 5, 100 - 5
ax_precision, ax_pdf_a, ax_pdf_b = np.array([0.] * size_node), np.array([0.] * size_node), np.array([0.] * size_node)

for l0 in range(i, size_node + 1):
    l0_ = l0 - i
    ax_pdf_a[l0 - 1] = compute_norm_level(l0_, size_node_, ax_bits, ax_factors, K - 1)
    ax_pdf_b[l0 - 1] = compute_norm_level(l0, size_node, ax_bits, ax_factors, K - 1)
    ax_precision[l0 - 1] = (pf * ax_pdf_a[l0 - 1]) / ((pf * ax_pdf_a[l0 - 1]) + ((1 - pf) * ax_pdf_b[l0 - 1]))

import IPython
IPython.embed()

fig_precision_4 = plt.figure()
plt.plot(np.array(range(1, size_node + 1)), ax_precision, 'k')
plt.plot(np.array(range(1, size_node + 1)), ax_pdf_a * 10., 'b')
plt.plot(np.array(range(1, size_node + 1)), ax_pdf_b * 10., 'r')
plt.title("Precision of node's levels with different overlap value")
plt.show()

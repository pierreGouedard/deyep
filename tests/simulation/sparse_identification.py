# Global import
import numpy as np
from matplotlib.pyplot import cm
from matplotlib.colors import rgb2hex

# Local import
from deyep.utils import test_signal as ts
from deyep.core.solver import sampler, drainer
from deyep.utils import interactive_plots as ip


"""
This script is meant to validate that simulated data meets theoretical expectation. More specifically, this script will 
run the simulation of the 'Signal plus Noise' model.
"""

# Core params of the simulation
p_targets, p_bits, n_targets, n_bits = 0.3, 0.3, 10, 1000

# drainer params
t, i = 1000, 0

# Pseudo random
np.random.seed(1234)

# Create simulation abd generate I/O
simu = ts.SparseActivation(p_targets, p_bits, n_targets, n_bits, purity_rank=10)\
    .build_map_targets_bits()
imputer, dirin, dirout = simu.stream_io_sequence(10000, return_dirs=True)

# init tracking of simulaiton
target, purity = 0, 10

# Sample and drain
smp = sampler.Sampler([simu.n_bits, simu.n_targets], simu.N(t, i), imputer)\
    .sample()\
    .build_graph_multiple_output(t)

# Drain
drn = drainer.FiringGraphDrainer(simu.rho(i), smp.firing_graph.copy(), imputer, depth=2, verbose=1)

# Get targets bits and
l_bits = list(smp.preselect_bits[target])
ax_bits = np.zeros((len(l_bits), 1))

# Get rank of preselected bits
d_rank = simu.get_ranks(target)
d_rank.update({j: -1 for j in set(l_bits).difference(d_rank.keys())})

# TODO: Build a correct visualisation
# from deyep.utils import interactive_plots as ip
# from deyep.core.firing_graph.utils import gather_matrices
# ip.plot_graph(gather_matrices(smp.firing_graph.I.toarray(), smp.firing_graph.D.toarray(), smp.firing_graph.O.toarray()))

import IPython
IPython.embed()

stop, j, fg = False, 0, None
while not stop:
    print 'Iteration {}'.format((j + 1) * 10)
    fg = drn.drain(10, early_stoping=True).firing_graph
    ax_bits = np.hstack((ax_bits, fg.Iw.toarray()[l_bits, target][:, np.newaxis]))

    stop = not fg.mask_mat['I'].toarray().any()
    j += 1

d_colored_series, d_cmap, cmap = {}, {}, cm.get_cmap('brg', 11)
for k, v in d_rank.items():
    if v not in d_cmap.keys():
        c = rgb2hex(cmap((v + 1) % 11))
        d_cmap[v] = c
    else:
        c = d_cmap[v]

    d_colored_series[v] = d_colored_series.get(v, []) + [ax_bits[l_bits.index(k), 1:]]

import IPython
IPython.embed()

ip.multi_line_plot_colored(
    d_colored_series,
    title=r'Score process for different vertice colored according to their rank',
    ylab=r'Value of the score process',
    xlab=r"Number of iteration ($\times 10$)",
    cmap=d_cmap
)

import IPython
IPython.embed()

l_target_bits = [k for k, v in d_rank.items() if v == 6]
ax_ins, ax_out = imputer.features_forward.toarray()[:, l_target_bits], imputer.features_backward.toarray()[:, 0]

activation_rate = 0.3 + (0.7 * (1 - pow(0.7, 5)))
phi = simu.phi(simu.omega(0, 6))

print 'Theoretical values: mean activation = {}, phi = {}'.format(activation_rate, phi)
for i in range(len(l_target_bits)):
    ax_in = ax_ins[:, i]
    n_good = ax_in.astype(float).dot(ax_out.astype(float))
    n_bad = ax_in.astype(float).dot(1 - ax_out.astype(float))
    est_activation_rate = float(n_good + n_bad) / 10000
    est_phi = float(n_good) / (n_good + n_bad)
    print 'estimated activation = {}, estimated phi = {}'.format(est_activation_rate, est_phi)


bitoi = 8 #18, 19

# Scalar product (good points)
ax_in[:100, bitoi].astype(float).dot(ax_out[:100].astype(float))

# Scalar perp product
ax_in[:100, bitoi].astype(float).dot(1. - ax_out[:100].astype(float))


# Remove tmpdir
dirin.remove(), dirout.remove()





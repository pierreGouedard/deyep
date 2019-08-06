# Global import
import numpy as np

# Local import
from deyep.utils import test_signal as ts
from deyep.core.solver import sampler, drainer
from deyep.utils import interactive_plots as ip
from tests.simulation import utils as u
"""
This script is meant to validate that simulated data meets theoretical expectation. More specifically, this script will 
run the simulation of the 'Signal plus Noise' model.
"""

# Global parameter
np.random.seed(12345)
t, n_bits, p_targets, p_bits, n_targets = 200, 1000, 0.2, 0.25, 10
target, resolution = 0, 10

# Run simulation

# Create simulation and get imputer
simu = ts.SparseActivation(p_targets, p_bits, n_targets, n_bits).build_map_targets_bits()
imputer, dirin, dirout = simu.stream_io_sequence(10000, return_dirs=True)

# Sample and drain
smp = sampler.Sampler([simu.n_bits, simu.n_targets], simu.N(t, 0), imputer).sample().build_graph_multiple_output(t)

# Drain
drn = drainer.FiringGraphDrainer(simu.rho(0), smp.firing_graph.copy(), imputer, depth=2, verbose=1)

# init tracking of simulaiton
l_ps_bits, l_target_bits = list(smp.preselect_bits[target]), list(simu.target_bits[target])
l_noisy_bits = list(set(l_ps_bits).difference(l_target_bits))
ax_noisy_bits, ax_target_bits = np.zeros((len(l_noisy_bits), 1)), np.zeros((len(l_target_bits), 1))

stop, i, fg = False, 0, None
while not stop:
    print 'Iteration {}'.format((i + 1) * resolution)
    fg = drn.drain(resolution, early_stoping=True).firing_graph
    ax_noisy_bits = np.hstack((ax_noisy_bits, fg.Iw.toarray()[l_noisy_bits, target][:, np.newaxis]))
    ax_target_bits = np.hstack((ax_target_bits, fg.Iw.toarray()[l_target_bits, target][:, np.newaxis]))

    stop = not fg.mask_mat['I'].toarray().any()
    i += 1

# Plot the result
import IPython
IPython.embed()

ip.multi_line_plot_colored(
    {'b': ax_target_bits[:, 1:], 'r': ax_noisy_bits[:, 1:],
     'k': np.ones((1, ax_target_bits.shape[1])) * simu.mean_score_signal(t, 0)},
    title=r'Score process for different vertice ($p_f=0.3$, $p_n=0.6$, $t=500$) ',
    ylab=r'Value of the score process',
    xlab=r"Number of iteration ($\times 10$)"
)

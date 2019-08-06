# Global import
import numpy as np

# Local import
from deyep.utils import test_signal as ts
from deyep.core.solver import sampler, drainer
from tests.simulation.utils import run_signal_plus_noise_simulation
from deyep.utils import interactive_plots as ip

"""
This script is meant to validate that simulated data meets theoretical expectation. More specifically, this script will 
run the simulation of the 'Signal plus Noise' model.
"""

# Global parameter
t, n_bits, p_noise, p_target, n_targets = 300, 1000, 0.5, 0.3, 50

d_scores = {
    'targets': {'x': np.arange(0, 12, 2), 'data': [], 'color': 'lightblue'},
    'noise': {'x': np.arange(0, 12, 2), 'data': [], 'color': 'pink'}
}

###################### TEST
np.random.seed(1234)
# Create simulation and get imputer
simu = ts.SignalPlusNoise(1, n_bits, p_target, n_targets, p_noise)
imputer, dirin, dirout = simu.stream_io_sequence(10000, return_dirs=True)

# Sample and drain
target_bits = {0: np.random.choice(simu.target_bits[0], 5, replace=False)}
smp = sampler.Sampler(
    [simu.n_sim * simu.n_bits, simu.n_sim], simu.N(t, 5), imputer, selected_bits=target_bits
).sample().build_graph_multiple_output(t)
drn = drainer.FiringGraphDrainer(simu.rho(5), smp.firing_graph.copy(), imputer, depth=3, verbose=1)

# Initialize experimental's trackers of information
l_ps_bits, l_target_bits = list(smp.preselect_bits[0]), list(simu.target_bits[0])
l_noisy_bits = list(set(l_ps_bits).difference(l_target_bits))
ax_noisy_bits, ax_target_bits = np.zeros((len(l_noisy_bits), 1)), np.zeros((len(l_target_bits), 1))

# Drain firing graph
stop, i = False, 0
while not stop:
    print 'Iteration {}'.format((i + 1) * 10)
    fg0_ = drn.drain(10).firing_graph

    ax_noisy_bits = np.hstack((ax_noisy_bits, fg0_.Iw.toarray()[l_noisy_bits, 0][:, np.newaxis]))
    ax_target_bits = np.hstack((ax_target_bits, fg0_.Iw.toarray()[l_target_bits, 0][:, np.newaxis]))

    stop = not fg0_.I_mask.toarray().any()
    i += 1

import IPython
IPython.embed()
# TODO: there is a fucking problem
ip.multi_line_plot_colored(
    {'b': ax_target_bits[:, 1:], 'r': ax_noisy_bits[:, 1:], 'k': np.ones((1, 100)) * simu.mean_score_signal(t, 5)},
    title=r'Score process for different vertice ($p_f=0.3$, $p_n=0.1$, $t=100$) ',
    ylab=r'Value of the score process',
    xlab=r"Number of iteration ($\times 10$)"
)
###################### TEST

for i in np.arange(0, 12, 2):
    ax_noise_bits_, ax_target_bits_ = run_signal_plus_noise_simulation(t, n_bits, p_noise, p_target, n_targets, i)

    # Add to dict containing scores
    d_scores['targets']['data'].append(ax_target_bits_)
    d_scores['noise']['data'].append(ax_noise_bits_)




# Buid Box plot
ip.multi_box_plot(
    d_scores,
    title=r"evolution of measure grid's bit of target factor for different $p_N$ ($p_f=0.3$, $t=100$)",
    ylab=r'Value of the score process',
    xlab=r"Values of $p_N$"
)








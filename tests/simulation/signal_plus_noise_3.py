# Global import
import numpy as np

# Local import
from deyep.utils import test_signal as ts
from deyep.core.solver import sampler, drainer
from tests.simulation.utils import run_signal_plus_noise_simulation
from deyep.utils import interactive_plots as ip
from tests.simulation import utils as u

"""
This script is meant to validate that simulated data meets theoretical expectation. More specifically, this script will 
run the simulation of the 'Signal plus Noise' model.
"""

# Global parameter
t, n_bits, p_noise, p_target, n_targets = 500, 1000, 0.6, 0.3, 50

d_scores = {
    'targets': {'x': np.arange(0, 12, 2), 'data': [], 'color': 'lightblue'},
    'noise': {'x': np.arange(0, 12, 2), 'data': [], 'color': 'pink'}
}


###################### TEST
n_selected = 5
ax_noisy_bits, ax_target_bits, simu = \
    u.run_signal_plus_noise_simulation(t, n_bits, p_noise, p_target, n_targets, n_selected)


# TODO: there is a fucking problem
ip.multi_line_plot_colored(
    {'b': ax_target_bits[:, 1:], 'r': ax_noisy_bits[:, 1:],
     'k': np.ones((1, ax_target_bits.shape[1])) * simu.mean_score_signal(t, 5)},
    title=r'Score process for different vertice ($p_f=0.3$, $p_n=0.6$, $t=500$) ',
    ylab=r'Value of the score process',
    xlab=r"Number of iteration ($\times 10$)"
)
###################### TEST

for i in np.arange(0, 6, 1):
    _, ax_target_bits_, _ = u.run_signal_plus_noise_simulation(t, n_bits, p_noise, p_target, n_targets, i)

    # Add to dict containing scores
    d_scores['targets']['data'].append(ax_target_bits_[:, 1:])
    d_scores['noise']['box_data'].append(ax_target_bits_[:, -1])

import IPython
IPython.embed()

# Buid Box plot
ip.multi_box_plot(
    d_scores,
    name_data='scores',
    title=r"evolution of measure grid's bit of target factor for different $p_N$ ($p_f=0.3$, $t=100$)",
    ylab=r'Value of the score process',
    xlab=r"Values of $p_N$"
)








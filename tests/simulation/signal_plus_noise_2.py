# Global import
import pickle
import string
import numpy as np

# Local import
from tests.simulation.utils import run_signal_plus_noise_simulation
from deyep.utils import interactive_plots as ip

"""
NOT SURE THIS VISUALISATION IS RELEVANT
This script is meant to validate that simulated data meets theoretical expectation. More specifically, this script will 
run the simulation of the 'Signal plus Noise' model.
"""

# Global parameter
n_bits, p_target, n_targets = 1000, 0.3, 50
l_t, l_p_noise = [300, 500, 700, 900], [0.3, 0.5, 0.7, 0.9]
d_scores = {
    'targets': {'x': l_p_noise, 'scores': [], 'data': [], 'color': 'lightblue'}
}

# Run the simulation
for t, p_noise in zip(l_t, l_p_noise):
    _, ax_target_bits_, _ = run_signal_plus_noise_simulation(
        t, n_bits, p_noise, p_target, n_targets, 0, resolution=10, verbose=1
    )

    # Add to dict containing scores
    d_scores['targets']['data'].append(ax_target_bits_[:, 1:])
    d_scores['targets']['scores'].append(ax_target_bits_[:, -1])

sim_name = ''.join(np.random.choice(list(string.ascii_letters), 10))
with open('sim_{}.pickle'.format(sim_name), 'wb') as handle:
    pickle.dump(d_scores, handle)

# Buid Box plot
# ip.multi_box_plot(
#     d_scores,
#     name_data='scores',
#     title=r"Evolution of measure grid's bit of target factor for different $p_N$ ($p_f=0.3$, $t=100$)",
#     ylab=r'Value of the score process',
#     xlab=r"Values of $p_N$"
# )








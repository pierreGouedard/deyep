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
t, n_bits, p_noise, p_target, n_targets = 500, 1000, 0.6, 0.3, 50

# Run simulation
ax_noisy_bits, ax_target_bits, simu = u.run_signal_plus_noise_simulation(t, n_bits, p_noise, p_target, n_targets, 0)

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

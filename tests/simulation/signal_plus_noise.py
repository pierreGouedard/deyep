# Global import


# Local import
from deyep.utils import test_signal as ts
from deyep.core.solver import sampler, drainer

"""
This script is meant to validate that simulated data meets theoretical bounds. More specifically, this script will run
a simulation of the 'Signal plus Noise' case.
"""

# Global parameter
t = 100
n_bits = 1000
n_sim = 10

# Target factor parameter
p_target = 0.3
n_targets = 50

# Noisy activation parameter
p_noise = 0.1

simu = ts.SignalPlusNoise(n_sim, n_bits, p_target, n_targets, p_noise)

# Generate signal
sax_in, sax_out = simu.generate_io_sequence(10000)

# Create imputer
imputer, dirin, dirout = simu.stream_io_sequence(10000, return_dirs=True)

################################################## Iteration 0 #########################################################
nu_0, rho_0, T_0, N_0 = simu.get_tuple(0, t)
import IPython
IPython.embed()

# Sample
smp = sampler.Sampler([simu.n_sim * simu.n_bits, simu.n_sim], N_0, imputer, selected_bits=None, preselected_bits=None)
fg0 = smp.sample().build_graph_multiple_output().firing_graph

# Drain
drn = drainer.FiringGraphDrainer(rho_0, fg0, imputer, depth=2, verbose=1, track_backward=True, track_forward=True)
fg0_ = drn.drain(t * 2).firing_graph

# Analysis of iteration 0
selected_bits, preselected_bits = None, None

################################################## Iteration 1 #########################################################

nu_1, rho_1, T_1, N_1 = simu.get_tuple(1, t)

# Sample
smp = sampler.Sampler(
    [simu.n_sim * simu.n_bits, simu.n_sim], N_0, imputer, selected_bits=selected_bits, preselected_bits=preselected_bits
)
fg1 = smp.sample().build_graph_multiple_output().firing_graph

# Drain
drn = drainer.FiringGraphDrainer(rho_1, fg1, imputer, depth=3, verbose=1, track_backward=True, track_forward=True)
fg1_ = drn.drain(t * 2).firing_graph





# Remove tmpdir
dirin.remove(), dirout.remove()




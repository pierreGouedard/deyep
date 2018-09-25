# Global import

# Local import
from deyep.utils.simulations import init_imputer
from deyep.core.builder.binomial import BinomialGraphBuilder
from deyep.utils.code_builder import CodeBuilder
from deyep.utils.driver.nmp import NumpyDriver
from deyep.core.imputer import identity
from deyep.core.solver.canonical import CanonicalDeepNetSolver
from deyep.core.deep_network import DeepNetwork
import settings

# Define base feature of simulation
simulation = 'code_breaker'
dir_in = settings.deyep_raw_path.format(simulation)
dir_out = settings.deyep_io_path.format(simulation)
driver = NumpyDriver()

# Create imputer paths if necessary
if not driver.exists(dir_in):
    driver.makedirs(dir_in)

if not driver.exists(dir_out):
    driver.makedirs(dir_out)

# Generate and store random code
n_i, n_o = 10, 5
code_builder = CodeBuilder(20, [n_i, n_o], p=0.5, seed=1234)\
    .generate_code()\
    .save_random_sequence(1000, dir_in, 'forward.npz', 'backwward.npz', driver, sparse=True)

# Set core deyep parameters
ni, nd, no, depth, p0, l0, tau, w0, basis, capacity, delay = n_i, 1000, n_o, 5, 0.1, 10, 5, 10, 'canonical', 10, 0

# Core element
imputer = init_imputer(identity.DoubleIdentityImputer(simulation, dir_in, dir_out))
network = BinomialGraphBuilder(ni, nd, no, depth, p0, w0, l0, tau, capacity).build_network(simulation, 1)

import IPython
IPython.embed()

# Buffer period: save imputer and network
imputer.save()
network.save()

# TODO: After this step initialize imputer and save both the networks and the imputer
import IPython
IPython.embed()

imputer = identity.DoubleIdentityImputer.load(simulation)
network = DeepNetwork.load(simulation, 1)

imputer.stream_features()
solver = CanonicalDeepNetSolver(network, delay, imputer)


# Fir solver
solver.fit_epoch(500)





# Global import
import sys
import numpy as np

# Local import
from deyep.core.builder.binomial import BinomialGraphBuilder
from deyep.utils.code_builder import CodeBuilder
from deyep.core.imputer import identity
from deyep.utils.simulations import Simulation


class CodeBreaker(Simulation):

    name = 'code_breaker'
    params_network = {'ni': 10, 'nd': 100, 'no': 5, 'depth': 2, 'p0': 0.1, 'l0': 10, 'tau': 5, 'w0': 10,
                      'basis': "canonical", 'capacity': 10, 'delay': 0}
    imputer = identity.DoubleIdentityImputer
    builder = BinomialGraphBuilder
    raw_builder = CodeBuilder(20, [10, 5], p=0.5, seed=1234).generate_code()
    n_network = 100

    def __init__(self, resume=False):
        Simulation.__init__(self, CodeBreaker.name, CodeBreaker.imputer, CodeBreaker.params_network,
                            CodeBreaker.builder, raw_builder=CodeBreaker.raw_builder, resume=resume,
                            n_network=CodeBreaker.n_network)

    def check_network_cleaning(self):
        # Fit network 0
        network_id = 0
        solver = self.fit_network(network_id=network_id, start_penalty=1, end_penalty=5, save_network=False)

        # Make sure the network cleaning does not change forward output of deep network
        ax_output = solver.transform_array(np.ones(self.params_network['ni']))
        solver.fit_epoch(200)
        solver.clean_network_nodes()
        ax_output_ = solver.transform_array(np.ones(self.params_network['ni']))

        assert (ax_output == ax_output_).all()

    def check_network_qualification(self):

        # Fit network 0
        network_id = 0
        _ = self.fit_network(network_id=network_id, start_penalty=1, end_penalty=20, save_network=False)

        # Qualify network 0
        self.qualify_network(network_id=network_id, size_test=1000)

        import IPython
        IPython.embed()


# TODO:
# Merge multiple networks


if __name__ == '__main__':
    l_args = []
    if len(sys.argv) > 1:
        l_args = sys.argv[1:]

    if 'resume' in l_args:
        sim = CodeBreaker(resume=True)
    else:
        sim = CodeBreaker()

    # Quick test of network cleaning: should be re-implemented in core unit tests
    #sim.check_network_cleaning()

    # Quick test of metrics of the fitted deep network
    sim.check_network_qualification()

    # Fit all network and visualize them (make sure the change in network is saved
    #sim.fit_all_networks(penalty_rate=400, start_penalty=1, end_penalty=10)

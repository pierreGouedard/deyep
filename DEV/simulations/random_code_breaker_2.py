# Global import
import sys
import numpy as np

# Local import
from deyep.core.builder.binomial import BinomialGraphBuilder
from deyep.utils.code_builder.simple_mapping import SimpleMapping
from deyep.core.imputer import identity
from deyep.utils.simulations import Simulation
from deyep.core.merger.comon import DeepNetMerger
from deyep.core.runner.comon import DeepNetRunner


class CodeBreaker(Simulation):

    name = 'code_breaker_2'
    params_network = {'ni': 10, 'nd': 100, 'no': 5, 'depth': 3, 'p0': 0.1, 'l0': 10, 'tau': 5, 'w0': 10,
                      'basis': "canonical", 'capacity': 5, 'delay': 1}
    imputer = identity.DoubleIdentityImputer
    builder = BinomialGraphBuilder
    raw_builder = SimpleMapping(20, [10, 5], p=0.5).generate_code()
    n_network = 1

    def __init__(self, resume=False):
        Simulation.__init__(self, CodeBreaker.name, CodeBreaker.imputer, CodeBreaker.params_network,
                            CodeBreaker.builder, raw_builder=CodeBreaker.raw_builder, resume=resume,
                            n_network=CodeBreaker.n_network)

    def check_network_cleaning(self):
        # Fit network 0 without any cleaning
        network_id = self.d_networks.values()[0].network_id
        solver = self.fit_network(network_id=network_id, start_penalty=1, end_penalty=5, save_network=False,
                                  clean=False)
        # save network
        deep_network_nasty = solver.deep_network

        # Clean network
        deep_network_cleaned = DeepNetMerger([solver.deep_network])\
            .clean_network() \
            .deep_network

        # Make sure cleaning didn't alter network fundamental structure
        ax_out_nasty = DeepNetRunner(deep_network_nasty, self.params_network['delay'] + 2, imputer=self.imputer)\
            .reset_runner()\
            .transform_array()
        ax_out_cleaned = DeepNetRunner(deep_network_cleaned, self.params_network['delay'] + 2, imputer=self.imputer)\
            .reset_runner()\
            .transform_array()

        # Print result
        print 'cleaning ok: {}'.format((ax_out_cleaned == ax_out_nasty).all())

        # Qualify network 0
        self.qualify_network(network_id=network_id, save_network=False)

        # Print result
        print 'Qualification of network: {}'.format(self.d_networks[network_id].d_metrics)

    def check_network_merge(self, n):
        #
        for nid, dn in self.d_networks.items():
            if not dn.is_fitted:
                self.fit_network(network_id=nid, start_penalty=1, end_penalty=20, rate_penalty=1, n_epoch=500,
                                 clean=False)

        dn_merged = self.merge_networks(self.d_networks.values())

        for _ in range(n):

            # Create new network
            d_networks = Simulation.create_networks(self.name, self.builder, 1, self.params_network)

            # Fit it
            solver = self.fit_network(
                network=d_networks.values()[0], start_penalty=1, end_penalty=20, rate_penalty=1, n_epoch=500,
                update=False, save_network=False, verbose=0, clean=False
            )

            # Merged it  with exisiting network
            if len(solver.deep_network.network_nodes) > 0:
                dn_merged = self.merge_networks([solver.deep_network, dn_merged], drop=True)

                # Print result qualification
                self.qualify_network(network=dn_merged)
                print 'Qualification of network: {}'.format(dn_merged.d_metrics)


if __name__ == '__main__':
    #TODO: Before launching run: export PYTHONPATH="/home/erepie/deyep/" => fix it
    l_args = []

    if len(sys.argv) > 1:
        l_args = sys.argv[1:]

    if 'resume' in l_args:
        sim = CodeBreaker(resume=True)
    else:
        sim = CodeBreaker()

    # Merge network
    sim.check_network_merge(100)

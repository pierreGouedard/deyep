# Global import
import sys

# Local import
from deyep.core.builder.binomial import BinomialGraphBuilder
from deyep.utils.code_builder.simple_mapping import SimpleMapping
from deyep.core.imputer import identity
from deyep.utils.simulations import Simulation
from deyep.core.merger.comon import DeepNetMerger
from deyep.core.runner.comon import DeepNetRunner


class CodeBreaker(Simulation):

    name = 'code_breaker'
    params_network = {'ni': 10, 'nd': 100, 'no': 5, 'depth': 2, 'p0': 0.1, 'l0': 10, 'tau': 5, 'w0': 10, 'capacity': 5,
                      'delay': 0}
    imputer = identity.DoubleIdentityImputer
    builder = BinomialGraphBuilder
    raw_builder = SimpleMapping(20, [10, 5], p=0.5).generate_code()
    n_network = 100

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

    def check_network_qualification(self):

        # Fit network 0
        network_id = self.d_networks.values()[0].network_id
        _ = self.fit_network(network_id=network_id, start_penalty=1, end_penalty=40, rate_penalty=2, save_network=False)

        # Qualify network 0
        self.qualify_network(network_id=network_id, save_network=False)

        # Print result
        print 'Qualification of network: {}'.format(self.d_networks[network_id].d_metrics)

    def check_network_merge(self):
        dn_merged = None

        for dn in list(self.d_networks.values()):
            solver = self.fit_network(network=dn,  start_penalty=1, end_penalty=40, rate_penalty=2, verbose=0)

            if len(solver.deep_network.network_nodes) > 0:
                print 'merging network {}'.format(dn.network_id)
                dn_merged = self.merge_networks([solver.deep_network, dn_merged], drop=True)

                # Print result qualification
                self.qualify_network(network=dn_merged)
                print 'Qualification of network: {}'.format(dn_merged.d_metrics)

            else:
                dn_ = self.d_networks.pop(dn.network_id)
                dn_.delete()


if __name__ == '__main__':

    #TODO: Before launching run: export PYTHONPATH="/home/erepie/deyep/" => fix it

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
    #sim.check_network_qualification()

    # Merge network
    #sim.check_network_merge()

    import IPython
    IPython.embed()

    # Fit all network and visualize them (make sure the change in network is saved
    #sim.fit_all_networks(penalty_rate=400, start_penalty=1, end_penalty=10)

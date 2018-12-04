# Global import
import string
import random
import numpy as np

# Local import
from deyep.core.builder.binomial import BinomialGraphBuilder
from deyep.core.solver.drainer import DeepNetDrainer
from deyep.core.solver.filler import DeepNetFiller
from deyep.core.solver.utils import DeepNetUtils
from deyep.core.runner.comon import DeepNetRunner


class DeepNetSolver(object):
    incompr_depth = 2

    def __init__(self, imputer, depth, params_network, deep_network=None, fill_method='binomial', start_penalty=1,
                 end_penalty=10, rate_penalty=1, size_epoch=200, verbose=1):

        # Core params deep network
        self.imputer, self.delay, self.depth, self.deep_network, self.params_network = \
            imputer, depth - DeepNetSolver.incompr_depth, depth, deep_network, params_network

        # Core param filler
        self.fill_method, self.fill_mapping, self.deep_network_fill, self.imputer_fill, self.main_mapping = \
            fill_method, None, None, None, {}

        # Core param global execution
        self.start_penalty, self.end_penalty, self.rate_penalty, self.size_epoch, self.verbose = \
            start_penalty, end_penalty, rate_penalty, size_epoch, verbose

    @staticmethod
    def rid(n):
        return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

    def build_network(self):
        # Instantiate builder
        builder = BinomialGraphBuilder(
            self.params_network['ni'], self.params_network['nd'], self.params_network['no'], self.depth,
            self.params_network['p0'], self.params_network['w0'], self.params_network['l0'], self.params_network['tau'],
            self.params_network['capacity']
        )

        # Create and set deep network
        self.deep_network = builder.build_network(
            self.params_network['name'], 'network_{}'.format(self.rid(7)), link_delay=self.delay
        )

    @staticmethod
    def set_imputer(imputer):
        imputer.read_raw_data('forward.npz', 'backward.npz')
        imputer.run_preprocessing()
        imputer.write_features('forward.npz', 'backward.npz')

        return imputer

    def display_stats(self, penalty):

        print 'penalty: {}'.format(penalty)
        self.qualify_network()
        print 'Qualification: {}'.format(self.deep_network.d_metrics)

        return self.deep_network.d_metrics

    def qualify_network(self):

        # Init imputer
        imputer = self.set_imputer(self.imputer.copy())
        imputer.stream_features(is_cyclic=False)

        # Get transformation of input
        runner = DeepNetRunner(self.deep_network, self.params_network['depth'], imputer)
        ax_out = runner.transform_array().toarray()

        # Set metrics of the deep network
        self.deep_network.set_metrics(self.depth, imputer.features_backward.toarray().astype(bool), ax_out)

    def shape_network_multi_penalties(self):
        # Make sure original deep network is built
        if self.deep_network is None:
            self.build_network()

        # Fit drainer
        for p in np.arange(self.start_penalty, self.end_penalty, self.rate_penalty):

            self.shape_network(p)
            stats = self.display_stats(p)

            # reset filler params
            self.fill_mapping, self.deep_network_fill, self.imputer_fill = None, None, None

            # quit if empty network
            if len(self.deep_network.network_nodes) == 0:
                break

            # Quit if maximum precision is reached
            if stats['P'] > 0.99:
                break

            # Display stats
            self.display_stats(p)

    def shape_network(self, penalty):
        depth = self.depth
        while depth > 0:
            if self.deep_network_fill is None:
                deep_network, imputer = self.deep_network.copy(), self.set_imputer(self.imputer.copy())
            else:
                deep_network, imputer = self.deep_network_fill.copy(), self.set_imputer(self.imputer_fill.copy())

            # Drain network
            deep_network = self.drain(deep_network, imputer, depth, penalty)

            # Merge network with current deep network
            self.clean_and_merge(deep_network)

            # Complete network with filling method
            if depth > 1:
                self.fill(deep_network, imputer, depth - 1)
                if self.imputer_fill is None:
                    break

            # decrement depth
            depth -= 1

        # clean network
        self.deep_network = DeepNetUtils(self.deep_network).clean_multi_side().deep_network

    def drain(self, dn, imputer, depth, penalty):
        # Init feature stream
        imputer.stream_features()

        # Drain network
        if depth > 1:
            drainer = DeepNetDrainer(dn, depth, imputer, p0=penalty, verbose=self.verbose - 1)
            drainer.fit_epoch(self.size_epoch)
            return drainer.deep_network.copy()

        else:
            return DeepNetDrainer.match(dn, imputer, penalty)

    def clean_and_merge(self, dn):
        dn_util = DeepNetUtils(dn)

        if self.deep_network_fill is None:
            self.deep_network = dn_util.clean_multi_side().deep_network

        else:
            dn_util = dn_util.combine(main=self.deep_network, d_map=self.fill_mapping).clean_multi_side()
            self.deep_network, self.main_mapping = dn_util.deep_network, dn_util.mapping_main_sub

        return dn_util.deep_network

    def fill(self, dn, imputer, depth):

        params_network_fill = {
            'ni': self.params_network['ni'], 'nd': self.params_network['nd'], 'depth': depth,
            'p0': self.params_network['p0'], 'w0': self.params_network['w0'], 'l0': self.params_network['l0'],
            'tau': self.params_network['tau'], 'capacity': self.params_network['capacity']
        }

        # Init feature stream
        imputer.stream_features()

        # Instantiate filler, create filling imputer and graph
        filler = DeepNetFiller(dn, imputer, params_network_fill)\
            .build_imputer_fill()\
            .build_graph_fill(BinomialGraphBuilder, self.rid(7), dry=False)

        self.deep_network_fill = filler.deep_network.copy()
        self.imputer_fill = filler.imputer.copy() if filler.imputer is not None else None
        self.fill_mapping = {self.main_mapping.get(k, k): v for k, v in filler.d_mapping.items()}

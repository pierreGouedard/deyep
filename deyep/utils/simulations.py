# Global import
import numpy as np
import string
import random

# Local import
import settings
from deyep.core.builder.comon import gather_matrices
from deyep.core.datastructures.deep_network import DeepNetwork
from deyep.core.solver.utils import DeepNetUtils
from deyep.core.runner.comon import DeepNetRunner
from deyep.core.solver.main import DeepNetSolver
from deyep.utils import interactive_plots as ip
from deyep.utils.driver.nmp import NumpyDriver


class Simulation(object):

    def __init__(self, name, imputer, params_network, builder, driver=NumpyDriver(), raw_builder=None,
                 sparse=True, resume=False):

        # Get admin information
        self.name = name
        self.dir_raw = settings.deyep_raw_path.format(name)
        self.dir_io = settings.deyep_io_path.format(name)
        self.dir_imp = settings.deyep_imputer_path.format(name)
        self.dir_dn = settings.deyep_network_path.format(name)
        self.dir_prod = settings.deyep_prod_path.format(name)
        self.driver = driver

        # Get network params
        self.builder = builder
        self.params_network = params_network
        self.params_network['delay'] = self.params_network['depth'] - DeepNetSolver.incompr_depth

        if resume:
            self.imputer = imputer.load(self.name)
            self.main_network = DeepNetwork.load(self.name, 'main')
            self.d_networks = {
                x.split('.')[0]: DeepNetwork.load(self.name, x.split('.')) for x in driver.listdir(self.name)
            }
        else:

            # Create dirs if needed
            Simulation.create_dirs(self.driver, self.dir_raw, self.dir_io, self.dir_dn, self.dir_imp, self.dir_prod)

            # build raw if necessary
            if raw_builder is not None:
                name_f, name_b = 'forward.{}'.format({True: 'npz'}.get(raw_builder.sparse, 'npy')), \
                    'backward.{}'.format({True: 'npz'}.get(raw_builder.sparse, 'npy'))
                raw_builder.save(self.dir_raw, name_f, name_b)

            # Create imputer
            self.imputer = Simulation.create_imputer(imputer(self.name, self.dir_raw, self.dir_io, is_sparse=sparse))

            # Create network
            self.main_network = \
                Simulation.create_networks(self.name, builder, 1, self.params_network, s_rid='main')['main']
            self.d_networks = {}

            # Save networks and imputer
            self.main_network.save()
            self.imputer.save()

    @staticmethod
    def create_dirs(driver, dir_raw, dir_io, dir_dn, dir_imp, dir_prod):
        if driver.exists(dir_raw):
            driver.remove(dir_raw, recursive=True)
        driver.makedirs(dir_raw)

        if driver.exists(dir_io):
            driver.remove(dir_io, recursive=True)
        driver.makedirs(dir_io)

        if driver.exists(dir_imp):
            driver.remove(dir_imp, recursive=True)
        driver.makedirs(dir_imp)

        if driver.exists(dir_dn):
            driver.remove(dir_dn, recursive=True)
        driver.makedirs(dir_dn)

        if driver.exists(dir_prod):
            driver.remove(dir_prod, recursive=True)
        driver.makedirs(dir_prod)

    @staticmethod
    def create_imputer(imputer):
        imputer.read_raw_data('forward.npz', 'backward.npz')
        imputer.run_preprocessing()
        imputer.write_features('forward.npz', 'backward.npz')

        return imputer

    @staticmethod
    def create_networks(name, builder, n_network, params_network, s_rid=None):

        builder = builder(
            params_network['ni'], params_network['nd'], params_network['no'], params_network['depth'],
            params_network['p0'], params_network['w0'], params_network['l0'], params_network['tau'],
            params_network['capacity']
        )

        def rid(n):
            return 'network_{}'.format(''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n)))

        d_networks = {}
        for _ in range(n_network):
            k = s_rid if s_rid is not None else rid(7)
            d_networks[k] = builder.build_network(name, k, link_delay=params_network['delay'])
            builder.reset_structure()

        return d_networks

    def add_networks(self, n):

        # Create  networs
        self.d_networks.update(Simulation.create_networks(self.name, self.builder, n, self.params_network))

        # Save networks and imputer
        for _, dn in self.d_networks.items():
            dn.save()

    def display_stats(self, dn, penalty):
        # Qualify network
        self.qualify_network(network=dn, save_network=False)

        # Clean it
        dnc = DeepNetMerger([dn]) \
            .clean_network() \
            .deep_network

        # display information
        print 'penalty: {}'.format(penalty)
        print 'Nb active node: {}'.format(len(dnc.network_nodes))
        print 'Qualification: {}'.format(dn.d_metrics)

        return dn.d_metrics

    def fit_network(self, network=None, size_epoch=100,  start_penalty=1, end_penalty=10, rate_penalty=1,
                    save_network=True, verbose=1):

        # Get network
        network_ = network if network is not None else self.main_network.copy()

        # Start features streaming
        imputer = Simulation.create_imputer(self.imputer.copy())
        imputer.stream_features()

        # Create and shape solver
        solver = DeepNetSolver(
            imputer, self.params_network['depth'], self.params_network, deep_network=network_,
            start_penalty=start_penalty, end_penalty=end_penalty, rate_penalty=rate_penalty, size_epoch=size_epoch,
            verbose=verbose
        )

        solver.shape_network_multi_penalties()

        # update state of network and replace
        solver.deep_network.is_fitted = True
        if network is None:
            self.main_network = solver.deep_network.copy()

            # Save fitted network
            if save_network:
                self.main_network.save()
        else:
            return solver.deep_network

    def qualify_network(self, network=None, save_network=True):

        # Get network
        network_ = network if network is not None else self.main_network.copy()

        # Init imputer
        imputer = Simulation.create_imputer(self.imputer.copy())
        imputer.stream_features(is_cyclic=False)

        # Get transformation of input
        runner = DeepNetRunner(network_, self.params_network['depth'], imputer)
        ax_out = runner.transform_array().toarray()

        # Set metrics of the deep network
        network.set_metrics(depth, imputer.features_backward.toarray().astype(bool), ax_out)

        # Save fitted network
        if network is None:
            self.main_network = network.copy()

            # Save fitted network
            if save_network:
                self.main_network.save()
        else:
            return network_

    def merge_network(self, dn=None, save=True, drop=True):

        dn_util = DeepNetUtils(self.main_network.copy())\
            .merge(dn)

        self.main_network = dn_util.deep_network

        if save:
            self.main_network.save()

        if drop:
            dn_ = self.d_networks.pop(dn.network_id, None)
            if dn_ is not None:
                dn_.delete()

    def visualize_network(self, network=None):

        # Get network
        network_ = network if network is not None else self.main_network.copy()

        ax_graph = gather_matrices(network_.I.toarray(), network_.D.toarray(), network_.O.toarray())
        ip.plot_graph(ax_graph, self.builder.layout(network_.I, network_.D, network_.O))





# Global import
import numpy as np
import string
import random

# Local import
import settings
from deyep.core.builder.comon import gather_matrices
from deyep.core.datastructures.deep_network import DeepNetwork
from deyep.core.merger.comon import DeepNetMerger
from deyep.core.runner.comon import DeepNetRunner
from deyep.core.solver.comon import DeepNetSolver
from deyep.utils import interactive_plots as ip
from deyep.utils.driver.nmp import NumpyDriver


class Simulation(object):

    def __init__(self, name, imputer, params_network, builder, n_network=1, driver=NumpyDriver(), raw_builder=None,
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
        self.n_networks = n_network
        self.params_network = params_network

        if resume:
            self.imputer = imputer.load(self.name)
            self.d_networks = {name.split('.')[0]: DeepNetwork.load(self.name, name.split('.')[0])
                               for name in driver.listdir(self.dir_dn)}
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

            # Create  networs
            self.d_networks = Simulation.create_networks(self.name, builder, self.n_networks, self.params_network)

            # Save networks and imputer
            self.save_networks()
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
    def create_networks(name, builder, n_network, params_network):

        builder = builder(params_network['ni'], params_network['nd'], params_network['no'],
                          params_network['depth'], params_network['p0'], params_network['w0'],
                          params_network['l0'], params_network['tau'], params_network['capacity'])

        def rid(n):
            return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

        d_networks = {}
        for _ in range(n_network):
            k = 'network_{}'.format(rid(7))
            d_networks[k] = builder.build_network(name, k, delay=params_network['delay'])
            builder.reset_structure()

        return d_networks

    def display_stats(self, dn, penalty):
        # Qualify network
        self.qualify_network(network=dn, update=False, save_network=False)

        # Clean it
        dnc = DeepNetMerger([dn]) \
            .clean_network() \
            .deep_network

        # display information
        print 'penalty: {}'.format(penalty)
        print 'Nb active node: {}'.format(len(dnc.network_nodes))
        print 'Qualification: {}'.format(dn.d_metrics)

        return dn.d_metrics

    def save_networks(self):
        for _, dn in self.d_networks.items():
            dn.save()

    def fit_network(self, network_id=None, network=None, n_epoch=200,  start_penalty=1, end_penalty=10, rate_penalty=1,
                    clean=True, update=True, save_network=True, verbose=1):

        if network is None:
            network = self.d_networks[network_id]
        else:
            network_id = network.network_id

        # Start features streaming
        imputer = Simulation.create_imputer(self.imputer.copy())
        imputer.stream_features()

        # Create solver
        solver = DeepNetSolver(network, self.params_network['delay'] + 2, imputer, verbose=verbose)

        # Fit solver
        for i in np.arange(start_penalty, end_penalty, rate_penalty):
            solver.p = i
            solver.fit_epoch(n_epoch)

            # Display stats
            stats = self.display_stats(solver.deep_network.copy(), i)

            # Clean network if specified
            if clean:
                solver.deep_network = DeepNetMerger([solver.deep_network])\
                    .clean_network()\
                    .deep_network
                solver.reset_solver(reset_imputer=True, reset_inputs=True, reset_time=True)
                print 'step {}: nb network nodes = {}'.format(i, len(solver.deep_network.network_nodes))

            # quit if empty network
            if len(solver.deep_network.network_nodes) == 0:
                break

            # Quit if maximum precision is reached
            if stats['P'] > 0.99:
                break

        # update state of network
        solver.deep_network.is_fitted = True

        # Save fitted network
        if save_network:
            solver.deep_network.save()

        # Update network of simulation
        if update:
            self.d_networks[network_id] = solver.deep_network

        return solver

    def fit_all_networks(self, n_epoch=200,  start_penalty=1, end_penalty=10, rate_penalty=1, force_fit=False,
                         verbose=1):
        for _, dn in self.d_networks.items():
            if not dn.is_fitted or force_fit:
                self.fit_network(network=dn, n_epoch=n_epoch,  start_penalty=start_penalty,
                                 end_penalty=end_penalty, rate_penalty=rate_penalty, verbose=verbose)

    def qualify_network(self, network=None, network_id=None, depth=1, update=True, save_network=True):

        if network is None:
            network = self.d_networks[network_id]
        else:
            network_id = network.network_id

        # Init imputer
        imputer = Simulation.create_imputer(self.imputer.copy())
        imputer.stream_features(is_cyclic=False)

        # Get transformation of input
        runner = DeepNetRunner(network, self.params_network['delay'] + 2, imputer)
        ax_out = runner.transform_array()

        # Set metrics of the deep network
        network.set_metrics(depth, imputer.features_backward.toarray().astype(bool), ax_out)

        # Save fitted network
        if save_network:
            network.save()

        # Update network of simulation
        if update:
            self.d_networks[network_id] = network

    def qualify_all_network(self, depth=1):
        for _, dn in self.d_networks.items():
            self.qualify_network(network=dn, depth=depth)

    def add_deep_networks(self, n):
        # Create  networs
        d_networks = Simulation.create_networks(self.name, self.builder, n, self.params_network)

        # Save networks and imputer
        for _, dn in d_networks.items():
            dn.save()

        self.d_networks.update(d_networks)

    def merge_networks(self, l_dn, drop=True, save=True):
        l_dn = filter(lambda dn: dn is not None, l_dn)
        l_dn = filter(lambda dn: len(dn.network_nodes) > 0, l_dn)

        if len(l_dn) == 0:
            return None

        elif len(l_dn) == 1:
            return l_dn[0]

        else:
            deep_network_merged = DeepNetMerger(l_dn)\
                .merge_network()\
                .deep_network

            self.d_networks[deep_network_merged.network_id] = deep_network_merged

            if save:
                deep_network_merged.save()

            if drop:
                for dn in l_dn:
                    dn_ = self.d_networks.pop(dn.network_id, None)
                    if dn_ is not None:
                        dn_.delete()

            return deep_network_merged

    def export_dry_deep_network(self):
        raise NotImplementedError

    def visualize_network(self, network_id=None, network=None):
        if network is None:
            network = self.d_networks[network_id]

        ax_graph = gather_matrices(network.I.toarray(), network.D.toarray(), network.O.toarray())
        ip.plot_graph(ax_graph, self.builder.layout(network.I, network.D, network.O))





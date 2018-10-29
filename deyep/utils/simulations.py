# Global import
import numpy as np

# Local import
from deyep.utils.driver.nmp import NumpyDriver
from deyep.core.deep_network import DeepNetwork
from deyep.core.solver.canonical import CanonicalDeepNetSolver
from deyep.core.solver.fourrier import FourrierDeepNetSolver
from deyep.utils import interactive_plots as ip
from deyep.core.builder.comon import gather_matrices
import settings


class Simulation(object):

    def __init__(self, name, imputer, params_network, builder, n_network=1, driver=NumpyDriver(), raw_builder=None,
                 sparse=True, resume=False):

        # Get admin information
        self.name = name
        self.dir_raw = settings.deyep_raw_path.format(name)
        self.dir_io = settings.deyep_io_path.format(name)
        self.dir_imp = settings.deyep_imputer_path.format(name)
        self.dir_dn = settings.deyep_network_path.format(name)
        self.driver = driver

        # Get network params
        self.builder = builder
        self.n_networks = n_network
        self.params_network = params_network

        if resume:
            self.imputer = imputer.load(self.name)
            self.l_networks = [DeepNetwork.load(self.name, 'network_{}'.format(i)) for i in range(self.n_networks)]
        else:

            # Create dirs if needed
            Simulation.create_dirs(self.driver, self.dir_raw, self.dir_io, self.dir_dn, self.dir_imp)

            # build raw if necessary
            if raw_builder is not None:
                name_f, name_b = 'forward.{}'.format({True: 'npz'}.get(raw_builder.sparse, 'npy')), \
                                 'backward.{}'.format({True: 'npz'}.get(raw_builder.sparse, 'npy'))
                raw_builder.save(self.dir_raw, name_f, name_b, self.driver)

            # Create imputer
            self.imputer = Simulation.create_imputer(imputer(self.name, self.dir_raw, self.dir_io, is_sparse=sparse))

            # Create  networs
            self.l_networks = Simulation.create_networks(self.name, builder, self.n_networks, self.params_network)

            # Save networks and imputer
            for dn in self.l_networks:
                dn.save()

            self.imputer.save()

    @staticmethod
    def create_dirs(driver, dir_raw, dir_io, dir_dn, dir_imp):
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

        l_networks = [builder.build_network(name, 'network_{}'.format(i)) for i in range(n_network)]

        return l_networks

    def fit_network(self, network_id=0, network=None, penalty_rate=200,  start_penalty=1, end_penalty=10,
                    clean=True, update=True, save_network=True):

        if network is None:
            network = self.l_networks[network_id]

        # Start features streaming
        self.imputer.stream_features()

        if self.params_network['basis'] == 'canonical':
            solver = CanonicalDeepNetSolver(network, self.params_network['delay'] + 2, self.imputer)
        else:
            raise NotImplementedError

        # Fit solver
        for i in np.arange(start_penalty, end_penalty):
            solver.p = i
            solver.fit_epoch(penalty_rate)
            if clean:
                solver.clean_network_nodes()

        # Save fitted network
        if save_network:
            solver.deep_network.save()

        # Update network of simulation
        if update:
            self.l_networks[network_id] = solver.deep_network

        return solver

    def fit_all_networks(self, penalty_rate=200,  start_penalty=1, end_penalty=10):
        for dn in self.l_networks:
            self.fit_network(network=dn, penalty_rate=penalty_rate,  start_penalty=start_penalty,
                             end_penalty=end_penalty)

    def qualify_network(self, network=None, network_id=None, depth=1, size_test=200, penalty=10):

        if network is None:
            network = self.l_networks[network_id]

        solver = self.fit_network(network=network, penalty_rate=size_test,  start_penalty=1,
                                  end_penalty=2, clean=False, update=False, save_network=False)

        # Get transformation of input
        ax_out = solver.transform_array(self.imputer.read_features().features_forward.toarray(),
                                        max_iter=self.params_network['delay'] + 1)
        ax_out = ax_out[self.params_network['delay'] + 1:]

        # Set metrics of the deep network
        network.set_metrics(depth, self.imputer.features_backward.toarray().astype(bool), ax_out)

    def qualify_all_network(self, depth=1):
        for dn in self.l_networks:
            self.qualify_network(network=dn, depth=depth)

    def visualize_network(self, network_id=0, network=None):
        if network is None:
            network = self.l_networks[network_id]

        ax_graph = gather_matrices(network.I.toarray(), network.D.toarray(), network.O.toarray())
        ip.plot_graph(ax_graph, self.builder.layout(network.I, network.D, network.O))





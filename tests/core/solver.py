# Global imports
import unittest

import numpy as np
from scipy.sparse import csc_matrix

from deyep.core.builder.comon import mat_from_tuples, gather_matrices
from deyep.core.datastructures.deep_network import DeepNetwork
from deyep.core.imputer.identity import DoubleIdentityImputer
from deyep.core.solver.comon import DeepNetSolver
from deyep.utils.driver.nmp import NumpyDriver
from deyep.utils.interactive_plots import plot_graph
from tests.utils import testPattern as tp

__maintainer__ = 'Pierre Gouedard'


class TestSolver(unittest.TestCase):
    def setUp(self):

        # enable, disable visual inspection of graph
        self.visual = False

        # Set core parameters
        self.l0, self.tau, self.wo = 10, 5, 10

        # Create And pattern
        self.n_and = 10
        self.and_pat = tp.AndPattern(self.n_and)
        mat_in, mat_net, mat_out = mat_from_tuples(
            self.and_pat.build_graph_pattern_init(), len(self.and_pat.input_nodes), len(self.and_pat.network_nodes),
            len(self.and_pat.output_nodes),  weights=[20] * len(self.and_pat.build_graph_pattern_init())
        )
        self.and_dn = DeepNetwork.from_matrices(
            'test_and', mat_net, mat_in, mat_out, 10, w0=self.wo, l0=self.l0, tau=self.tau
        )

        # Create Xor pattern
        self.n_xor = 10
        self.xor_pat = tp.XorPattern(self.n_xor, self.n_xor)
        mat_in, mat_net, mat_out = mat_from_tuples(
            self.xor_pat.build_graph_pattern_init(), len(self.xor_pat.input_nodes), len(self.xor_pat.network_nodes),
            len(self.xor_pat.output_nodes), weights=[10] * len(self.xor_pat.build_graph_pattern_init())
        )
        self.xor_dn = DeepNetwork.from_matrices(
            'test_xor', mat_net, mat_in, mat_out, 10, w0=self.wo, l0=self.l0, tau=self.tau
        )

        mat_in, mat_net, mat_out = mat_from_tuples(
            self.xor_pat.build_graph_pattern_final(), len(self.xor_pat.input_nodes), len(self.xor_pat.network_nodes),
            len(self.xor_pat.output_nodes), weights=[10] * len(self.xor_pat.build_graph_pattern_init())
        )
        self.xor_dn_f = DeepNetwork.from_matrices(
            'test_xor_final', mat_net, mat_in, mat_out, 100, w0=self.wo, l0=self.l0, tau=self.tau
        )

        # Create Tree patter
        self.tree_pat = tp.TreePattern(5, 2)
        mat_in, mat_net, mat_out = mat_from_tuples(
            self.tree_pat.build_graph_pattern_init(), len(self.tree_pat.input_nodes), len(self.tree_pat.network_nodes),
            len(self.tree_pat.output_nodes), weights=[10] * len(self.tree_pat.build_graph_pattern_init())
        )
        self.tree_dn = DeepNetwork.from_matrices(
            'test_tree', mat_net, mat_in, mat_out, 10, w0=self.wo, l0=self.l0, tau=self.tau
        )

        mat_in, mat_net, mat_out = mat_from_tuples(
            self.tree_pat.build_graph_pattern_final(), len(self.tree_pat.input_nodes), len(self.tree_pat.network_nodes),
            len(self.tree_pat.output_nodes), weights=[10] * len(self.tree_pat.build_graph_pattern_final())
        )
        self.tree_dn_f = DeepNetwork.from_matrices(
            'test_tree_final', mat_net, mat_in, mat_out, 10, w0=self.wo, l0=self.l0, tau=self.tau
        )

    def andpattern(self):
        """
        python -m unittest tests.core.solver.TestSolver.andpattern

        """
        # Create temporary directory for test
        driver = NumpyDriver()
        tmpdirin, tmpdirout = driver.TempDir('test_and_pattern', suffix='in', create=True),  \
                              driver.TempDir('test_and_pattern', suffix='out', create=True)

        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.and_pat.generate_io_sequence(1000, seed=1234)
        driver.write_file(csc_matrix(ax_input), driver.join(tmpdirin.path, 'forward.npz'), is_sparse=True)
        driver.write_file(csc_matrix(ax_output), driver.join(tmpdirin.path, 'backward.npz'), is_sparse=True)

        # Create and init imputer
        imputer = DoubleIdentityImputer('test', tmpdirin.path, tmpdirout.path)
        imputer.read_raw_data('forward.npz', 'backward.npz')
        imputer.run_preprocessing()
        imputer.write_features('forward.npz', 'backward.npz')
        imputer.stream_features()

        # Create solver
        solver = DeepNetSolver(self.and_dn, self.and_pat.delay, imputer, p0=1)
        solver.fit_epoch(n=500)

        # Assert result is as expected
        self.assertTrue(all([solver.deep_network.network_nodes[0].d_levels[i] < self.tau for i in range(self.n_and)]))
        self.assertTrue(solver.deep_network.network_nodes[0].d_levels[10] > self.tau)

        # remove tmpdirs
        tmpdirin.remove(), tmpdirout.remove()

        # VISUAL TEST:
        if self.visual:
            ax_graph_conv = gather_matrices(self.and_dn.Iw.toarray(), self.and_dn.Dw.toarray(), self.and_dn.Ow.toarray())
            plot_graph(ax_graph_conv, self.and_pat.layout())

    def xorpattern(self):
        """
        python -m unittest tests.core.solver.TestSolver.xorpattern

        """

        # Create temporary directory for test
        driver = NumpyDriver()
        tmpdirin, tmpdirout = driver.TempDir('xorpattern', suffix='in', create=True),  \
                              driver.TempDir('xorpattern', suffix='out', create=True)

        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.xor_pat.generate_io_sequence(1000)

        driver.write_file(csc_matrix(ax_input), driver.join(tmpdirin.path, 'forward.npz'), is_sparse=True)
        driver.write_file(csc_matrix(ax_output), driver.join(tmpdirin.path, 'backward.npz'), is_sparse=True)

        # Create and init imputer
        imputer = DoubleIdentityImputer('test', tmpdirin.path, tmpdirout.path)
        imputer.read_raw_data('forward.npz', 'backward.npz')
        imputer.run_preprocessing()
        imputer.write_features('forward.npz', 'backward.npz')
        imputer.stream_features()

        # Create solver
        self.xor_dn.graph['Ow'] *= 10
        solver = DeepNetSolver(self.xor_dn, self.xor_pat.delay, imputer, p0=1)
        solver.fit_epoch(n=800)

        self.assertTrue(all([solver.deep_network.network_nodes[0].d_levels[i] < self.tau for i in range(self.n_xor)]))
        self.assertTrue(all([solver.deep_network.network_nodes[1].d_levels[i] < self.tau for i in range(self.n_xor)]))
        self.assertTrue((solver.deep_network.Iw.toarray()[:self.n_xor, 0] > 0).all())
        self.assertTrue((solver.deep_network.Iw.toarray()[self.n_xor:, 0] == 0).all())
        self.assertTrue((solver.deep_network.Iw.toarray()[:self.n_xor, 1] == 0).all())
        self.assertTrue((solver.deep_network.Iw.toarray()[self.n_xor:, 1] > 0).all())

        # remove tmpdirs
        tmpdirin.remove(), tmpdirout.remove()

        # VISUAL TEST:
        if self.visual:
            # Resulting network
            ax_graph = gather_matrices(self.xor_dn.Iw.toarray(), self.xor_dn.Dw.toarray(), self.xor_dn.Ow.toarray())
            plot_graph(ax_graph, self.xor_pat.layout(), 'Resulting graph test xor')

            # Expected network
            ax_graph = gather_matrices(self.xor_dn_f.Iw.toarray(), self.xor_dn_f.Dw.toarray(),
                                       self.xor_dn_f.Ow.toarray())
            plot_graph(ax_graph, self.xor_pat.layout(), 'Expecting graph test xor')

    def treepattern(self):
        """
        python -m unittest tests.core.solver.TestSolver.treepattern

        """

        # Create temporary directory for test
        driver = NumpyDriver()
        tmpdirin, tmpdirout = driver.TempDir('treepattern', suffix='in', create=True),  \
                              driver.TempDir('treepattern', suffix='out', create=True)

        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.tree_pat.generate_io_sequence(1000)
        driver.write_file(csc_matrix(ax_input), driver.join(tmpdirin.path, 'forward.npz'), is_sparse=True)
        driver.write_file(csc_matrix(ax_output), driver.join(tmpdirin.path, 'backward.npz'), is_sparse=True)

        # Create and init imputer
        imputer = DoubleIdentityImputer('test', tmpdirin.path, tmpdirout.path)
        imputer.read_raw_data('forward.npz', 'backward.npz')
        imputer.run_preprocessing()
        imputer.write_features('forward.npz', 'backward.npz')
        imputer.stream_features()

        # Create solver
        solver = DeepNetSolver(self.tree_dn, self.tree_pat.delay, imputer, p0=1)
        l_t_epoch, l_t_forwardp, l_t_forwardt, l_t_backwardp, l_t_backwardt = solver.epoch_analysis(100)

        print 'Mean running time of epoch: {} seconds'.format(np.mean(l_t_epoch))
        print 'Mean running time of forward transmit: {} seconds'.format(np.mean(l_t_forwardt))
        print 'Mean running time of backward transmit: {} seconds'.format(np.mean(l_t_backwardt))
        print 'Mean running time of forward process: {} seconds'.format(np.mean(l_t_forwardp))
        print 'Mean running time of backward process: {} seconds'.format(np.mean(l_t_backwardp))

        for i, j in zip(*self.tree_dn_f.O.nonzero()):
            self.assertTrue(solver.deep_network.Ow[i, j] > 0)

        # remove tmpdirs
        tmpdirin.remove(), tmpdirout.remove()

        # VISUAL TEST:
        if self.visual:
            # Resulting graph
            ax_graph = gather_matrices(self.tree_dn.Iw.toarray(), self.tree_dn.Dw.toarray(), self.tree_dn.Ow.toarray())
            plot_graph(ax_graph, self.tree_pat.layout(ax_graph))

            # Expected graph
            ax_graph = gather_matrices(self.tree_dn_f.Iw.toarray(), self.tree_dn_f.Dw.toarray(),
                                       self.tree_dn_f.Ow.toarray())
            plot_graph(ax_graph, self.tree_pat.layout(ax_graph))

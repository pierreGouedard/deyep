# Global imports
import unittest
import numpy as np
from scipy.sparse import csc_matrix

# Local import
from tests.utils import testPattern as tp
from deyep.core.solver.comon import DeepNetSolver
from deyep.core.builder.comon import mat_from_tuples, gather_matrices
from deyep.utils.interactive_plots import plot_graph, plot_hist
from deyep.core.deep_network import DeepNetwork
from deyep.utils.driver.nmp import NumpyDriver
from deyep.core.imputer.identity import DoubleIdentityImputer
__maintainer__ = 'Pierre Gouedard'


class TestSolverConvergence(unittest.TestCase):
    def setUp(self):

        # Set core parameters
        self.l0, self.tau, self.wo = 1000, 5, 10

        # Create random pattern with canonical basis
        self.rand_pat = tp.RandomPattern(10, 100, 10, 2, 0.6)

        mat_in, mat_net, mat_out = mat_from_tuples(self.rand_pat.build_graph_pattern_init(), self.rand_pat.ni,
                                                   self.rand_pat.nd, self.rand_pat.no,
                                                   weights=[10] * len(self.rand_pat.build_graph_pattern_init()))

        self.rand_dn_c = DeepNetwork.from_matrices(mat_net, mat_in, mat_out, 10, 'canonical', w0=self.wo, l0=self.l0,
                                                   tau=self.tau)

        # Create random pattern with Fourrier basis
        mat_in, mat_net, mat_out = mat_from_tuples(self.rand_pat.build_graph_pattern_init(), self.rand_pat.ni,
                                                   self.rand_pat.nd, self.rand_pat.no,
                                                   weights=[10] * len(self.rand_pat.build_graph_pattern_init()))
        self.rand_dn_f = DeepNetwork.from_matrices(mat_net, mat_in, mat_out, 100, 'fourrier', w0=self.wo, l0=self.l0,
                                                   tau=self.tau)

    def test_random_pattern_canonical_low_penalty(self):
        """
        python -m unittest tests.core.solver.convergence.TestSolverConvergence.test_random_pattern_canonical_low_penalty

        """
        # Create temporary directory for test
        driver = NumpyDriver()
        tmpdirin, tmpdirout = driver.TempDir('test_and_pattern', suffix='in', create=True),  \
                              driver.TempDir('test_and_pattern', suffix='out', create=True)

        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.rand_pat.generate_io_sequence(1000, seed=1234)
        driver.write_file(csc_matrix(ax_input), driver.join(tmpdirin.path, 'forward.npz'), is_sparse=True)
        driver.write_file(csc_matrix(ax_output), driver.join(tmpdirin.path, 'backward.npz'), is_sparse=True)

        # Create and init imputer
        imputer = DoubleIdentityImputer('test', tmpdirin.path, tmpdirout.path)
        imputer.read_raw_data('forward.npz', 'backward.npz')
        imputer.run_preprocessing()
        imputer.write_features('forward.npz', 'backward.npz')
        imputer.stream_features()

        # Create solver and run epoch
        solver = DeepNetSolver(self.rand_dn_c, self.rand_pat.delay, imputer, 'canonical', p0=1.)
        solver.run_epoch(500)

        Ow = solver.deep_network.Ow
        ax_values = np.array([Ow[i, j] for i, j in zip(*Ow.nonzero()) if Ow[i, j] < 20])
        import IPython
        IPython.embed()
        plot_hist(ax_values)

    def test_random_pattern_canonical_high_penalty(self):
        """
        python -m unittest tests.core.solver.convergence.TestSolverConvergence.test_random_pattern_canonical_high_penalty

        """
        # Create temporary directory for test
        driver = NumpyDriver()
        tmpdirin, tmpdirout = driver.TempDir('test_and_pattern', suffix='in', create=True),  \
                              driver.TempDir('test_and_pattern', suffix='out', create=True)

        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.rand_pat.generate_io_sequence(1000, seed=1234)
        driver.write_file(csc_matrix(ax_input), driver.join(tmpdirin.path, 'forward.npz'), is_sparse=True)
        driver.write_file(csc_matrix(ax_output), driver.join(tmpdirin.path, 'backward.npz'), is_sparse=True)

        # Create and init imputer
        imputer = DoubleIdentityImputer('test', tmpdirin.path, tmpdirout.path)
        imputer.read_raw_data('forward.npz', 'backward.npz')
        imputer.run_preprocessing()
        imputer.write_features('forward.npz', 'backward.npz')
        imputer.stream_features()

        # Create solver and run epoch
        solver = DeepNetSolver(self.rand_dn_c, self.rand_pat.delay, imputer, 'canonical', p0=10)
        solver.run_epoch(500)

        import IPython
        IPython.embed()
        Ow = solver.deep_network.Ow
        ax_values = np.array([Ow[i, j] for i, j in zip(*Ow.nonzero()) if Ow[i, j] < 20])


        plot_hist(ax_values)


        # Graph before execution
        ax_graph_conv = gather_matrices(self.rand_dn_c.I.toarray(), self.rand_dn_c.D.toarray(), self.rand_dn_c.O.toarray())
        plot_graph(ax_graph_conv, self.rand_pat.layout(ax_graph_conv))


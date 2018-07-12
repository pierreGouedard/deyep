# Global imports
import unittest
import numpy as np
from scipy.sparse import csc_matrix

# Local import
from tests.utils import testPattern as tp
from deyep.core.solver.comon import DeepNetSolver
from deyep.core.builder.comon import mat_from_tuples, gather_matrices
from deyep.utils.interactive_plots import plot_graph
from deyep.core.deep_network import DeepNetwork
from deyep.utils.driver.nmp import NumpyDriver
from deyep.core.imputer.identity import DoubleIdentityImputer
__maintainer__ = 'Pierre Gouedard'


class TestSolver(unittest.TestCase):
    def setUp(self):

        # Create And pattern
        self.and_pat = tp.AndPattern(10)
        mat_in, mat_net, mat_out = mat_from_tuples(self.and_pat.build_graph_pattern(), len(self.and_pat.input_nodes),
                                                   len(self.and_pat.network_nodes), len(self.and_pat.output_nodes))
        self.and_dn = DeepNetwork.from_matrices(mat_net, mat_in, mat_out, 100, w0=10, l0=10)

        # Create Xor pattern
        self.xor_pat = tp.XorPattern(10, 10)
        mat_in, mat_net, mat_out = mat_from_tuples(self.xor_pat.build_graph_pattern(), len(self.xor_pat.input_nodes),
                                                   len(self.xor_pat.network_nodes), len(self.xor_pat.output_nodes))
        self.xor_dn = DeepNetwork.from_matrices(mat_net, mat_in, mat_out, 100)

        # Create Tree patter
        self.tree_pat = tp.TreePattern(5, 3)
        mat_in, mat_net, mat_out = mat_from_tuples(self.tree_pat.build_graph_pattern(), len(self.tree_pat.input_nodes),
                                                   len(self.tree_pat.network_nodes), len(self.tree_pat.output_nodes))
        self.tree_dn = DeepNetwork.from_matrices(mat_net, mat_in, mat_out, 100)

    def test_and_pattern(self):
        """
        python -m unittest tests.core.solver.TestSolver.test_and_pattern

        """
        # Create temporary directory for test
        driver = NumpyDriver()
        tmpdirin, tmpdirout = driver.TempDir('test_and_pattern', suffix='in', create=True),  \
                              driver.TempDir('test_and_pattern', suffix='out', create=True)

        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.and_pat.generate_io_sequence(1000)
        driver.write_file(csc_matrix(ax_input), driver.join(tmpdirin.path, 'forward.npz'), is_sparse=True)
        driver.write_file(csc_matrix(ax_output), driver.join(tmpdirin.path, 'backward.npz'), is_sparse=True)

        # Create and init imputer
        imputer = DoubleIdentityImputer('test', tmpdirin.path, tmpdirout.path)
        imputer.read_raw_data('forward.npz', 'backward.npz')
        imputer.run_preprocessing()
        imputer.write_features('forward.npz', 'backward.npz')
        imputer.stream_features()

        # Create solver
        solver = DeepNetSolver(self.and_dn, self.and_pat.delay, imputer, 1)

        # TO TEST: interactive plot
        ax_graph = gather_matrices(self.and_dn.Iw.toarray(), self.and_dn.Dw.toarray(), self.and_dn.Ow.toarray())
        plot_graph(ax_graph)

        import IPython
        IPython.embed()



    def test_xor_pattern(self):
        """
        python -m unittest tests.core.equations.TestEquations.test_forward_transmiting

        """

    def test_tree_pattern(self):
        """
        python -m unittest tests.core.equations.TestEquations.test_forward_transmiting

        """


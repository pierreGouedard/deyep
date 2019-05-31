# Global imports
import unittest
import numpy as np
from scipy.sparse import csc_matrix

# Local import
from deyep.core.imputer.array import DoubleArrayImputer
from deyep.core.solver.sampler import Sampler
from deyep.utils.driver.nmp import NumpyDriver

__maintainer__ = 'Pierre Gouedard'


class TestDrainer(unittest.TestCase):
    def setUp(self):

        # enable, disable visual inspection of graph
        self.visual = False

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

    def andpattern(self):
        """
        python -m unittest tests.core.drainer.TestDrainer.andpattern

        """
        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.and_pat.generate_io_sequence(1000, seed=1234)
        imputer = create_imputer('andpattern', csc_matrix(ax_input), csc_matrix(ax_output))

        # Create drainer
        drainer = DeepNetDrainer(self.and_dn, self.and_pat.delay, imputer, p0=1)
        drainer.fit_epoch(n=500)

        # Assert result is as expected
        self.assertTrue(all([drainer.deep_network.network_nodes[0].d_levels[i] < self.tau for i in range(self.n_and)]))
        self.assertTrue(drainer.deep_network.network_nodes[0].d_levels[10] > self.tau)

        # VISUAL TEST:
        if self.visual:
            ax_graph_conv = gather_matrices(self.and_dn.Iw.toarray(), self.and_dn.Dw.toarray(), self.and_dn.Ow.toarray())
            plot_graph(ax_graph_conv, self.and_pat.layout())

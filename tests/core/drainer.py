# Global imports
import unittest
import numpy as np
from scipy.sparse import csc_matrix

# Local import
from deyep.core.firing_graph.utils import gather_matrices
from deyep.core.firing_graph.graph import FiringGraph
from deyep.core.imputer.array import DoubleArrayImputer
from deyep.core.solver.drainer import FiringGraphDrainer
from deyep.utils.interactive_plots import plot_graph
from tests.utils.testModel import AndPattern2 as ap2#, AndPattern3 as ap3

__maintainer__ = 'Pierre Gouedard'


class TestDrainer(unittest.TestCase):
    def setUp(self):

        # enable, disable visual inspection of graph
        self.visual = False

        # Create And pattern of depth 2
        self.ni, self.no = 10, 2
        self.ap2 = ap2(self.ni, self.no)
        self.ap2_fg = self.ap2.build_graph_pattern_init()

        # Create And pattern of depth 3
        # self.ni, self.no = 10, 2
        # self.ap3 = ap3(self.ni, self.no)
        # self.ap3_fg = self.ap3.build_graph_pattern_init()

    def andpattern(self):
            """
            python -m unittest tests.core.drainer.TestDrainer.andpattern

            """
            # Create I/O and save it into tmpdir files
            ax_input, ax_output = self.ap2.generate_io_sequence(1000, seed=1234)
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

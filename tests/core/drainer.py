# Global imports
import unittest
from scipy.sparse import csc_matrix
import numpy as np

# Local import
from deyep.core.firing_graph.utils import gather_matrices
from deyep.core.imputer.array import DoubleArrayImputer
from deyep.utils.driver.nmp import NumpyDriver
from deyep.core.solver.drainer import FiringGraphDrainer
from deyep.utils.interactive_plots import plot_graph
from deyep.utils.test_pattern import AndPattern2 as ap2, AndPattern3 as ap3

__maintainer__ = 'Pierre Gouedard'


class TestDrainer(unittest.TestCase):
    def setUp(self):

        # enable, disable visual inspection of graph
        self.visual = False

        # Create And pattern of depth 2 /!\ Do not change those parameter for the test /!\
        self.n, self.ni, self.no, self.w0, self.t_mask = 100, 10, 2, 10, 5
        self.ap2 = ap2(self.ni, self.no, w=self.w0, seed=1234)
        self.ap2_fg = self.ap2.build_graph_pattern_init()

        # Create And pattern of depth 3
        # self.ni, self.no, self.n_selected = 15, 2, 3
        # self.ap3 = ap3(self.ni, self.no, n_selected=self.n_selected, w=self.w0, seed=1234)
        # self.ap3_fg = self.ap3.build_graph_pattern_init()

    def time_mask(self):

        """
        Test the well functioning of mask on backward updates
        python -m unittest tests.core.drainer.TestDrainer.time_mask

        """

        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.ap2.generate_io_sequence(1000, seed=1234)
        imputer = create_imputer('andpattern2', csc_matrix(ax_input), csc_matrix(ax_output))

        # Create drainer
        drainer = FiringGraphDrainer(self.t_mask, 1, 1, self.ap2_fg, imputer, verbose=1)
        drainer.drain(100)

        # Get matrice of the graph
        fg, fg_final, fg_init = drainer.firing_graph, self.ap2.build_graph_pattern_final(), self.ap2.build_graph_pattern_init()
        I, I_init, I_final = fg.Iw.toarray(), fg_init.Iw.toarray(), fg_final.Iw.toarray()

        # Assert mask are working (no more than self.t_mask structure update
        self.assertTrue((I[I >= self.w0] == I_final[I_final > 0] + self.t_mask).all())
        self.assertTrue((I[(0 < I) & (I <= self.w0)] == I_init[~(I_final > 0) & (I_init > 0)] - self.t_mask).all())

    def andpattern2(self):

        """
        Test And Pattern of depth 2
        python -m unittest tests.core.drainer.TestDrainer.andpattern2

        """

        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.ap2.generate_io_sequence(1000, seed=1234)
        imputer = create_imputer('andpattern2', csc_matrix(ax_input), csc_matrix(ax_output))

        # Create drainer
        drainer = FiringGraphDrainer(1000, 1, 10, self.ap2_fg, imputer, verbose=1)
        drainer.drain(n=10)#self.n)

        # Get Data and assert result is as expected
        model_fg, I = self.ap2.build_graph_pattern_final(), drainer.firing_graph.Iw
        track_if = drainer.firing_graph.forward_firing['i']
        track_ib = drainer.firing_graph.backward_firing['i']

        # Check correctness of structure
        self.assertTrue((drainer.firing_graph.I.toarray() == model_fg.I.toarray()).all())

        # Test firing tracker
        self.assertTrue(all([I[j, 0] == track_ib[j, 0] + self.w0 for j in self.ap2.target[0]]))
        self.assertTrue(all([I[j, 1] == track_ib[j, 1] + self.w0 for j in self.ap2.target[1]]))
        self.assertTrue(all([(-3 < track_ib[j, 0] - track_if[0, j] <= 0) for j in self.ap2.target[0]]))
        self.assertTrue(all([(-3 < track_ib[j, 1] - track_if[0, j] <= 0) for j in self.ap2.target[1]]))

        # TODO: don't understand the difference between tracking backward and the actual value of edge's link
        import IPython
        IPython.embed()

        # VISUAL TEST:
        if self.visual:
            # GOT
            fg_got = self.ap2.build_graph_pattern_final()
            ax_graph_got = gather_matrices(fg_got.Iw.toarray(), fg_got.Dw.toarray(), fg_got.Ow.toarray())
            plot_graph(ax_graph_got, self.ap2.layout(), title='GOT')

            # Fring Graph at convergence
            ax_graph_conv = gather_matrices(
                self.ap2_fg.Iw.toarray(), self.ap2_fg.Dw.toarray(), self.ap2_fg.Ow.toarray()
            )

            plot_graph(ax_graph_conv, self.ap2.layout(), title='Result Test')

    def andpattern3(self):
        """
        Test And Pattern of depth 3
        python -m unittest tests.core.drainer.TestDrainer.andpattern3

        """
        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.ap3.generate_io_sequence(1000, seed=1234)
        imputer = create_imputer('andpattern3', csc_matrix(ax_input), csc_matrix(ax_output))

        # Create drainer
        drainer = FiringGraphDrainer(1, self.ap3_fg, imputer, depth=self.ap3.depth, verbose=1)
        drainer.drain(n=self.n * 2)

        # Get Data and assert result is as expected
        model_fg, I = self.ap3.build_graph_pattern_final(), drainer.firing_graph.Iw
        track_if = drainer.firing_graph.forward_firing['i']
        track_ib = drainer.firing_graph.backward_firing['ip'] + drainer.firing_graph.backward_firing['in']

        self.assertTrue((drainer.firing_graph.I.toarray() == model_fg.I.toarray()).all())

        # Those test could be done for 0, ..., no-1
        self.assertTrue(
            all([I[j, 1] == track_ib[j, 1] + self.w0 for j in self.ap3.target[0]
                 if j not in self.ap3.target_selected[0]])
        )
        self.assertTrue(
            all([I[j, 4] == track_ib[j, 4] + self.w0 for j in self.ap3.target[1]
                 if j not in self.ap3.target_selected[1]])
        )
        self.assertTrue(
            all([(-3 < track_ib[j, 1] - track_if[0, j] < 0) for j in self.ap3.target[0]
                 if j not in self.ap3.target_selected[0]])
        )
        self.assertTrue(
            all([(-3 < track_ib[j, 4] - track_if[0, j] < 0) for j in self.ap3.target[1]
                 if j not in self.ap3.target_selected[1]])
        )

        # VISUAL TEST:
        if self.visual:
            # GOT
            fg_got = self.ap3.build_graph_pattern_final()
            ax_graph_got = gather_matrices(fg_got.Iw.toarray(), fg_got.Dw.toarray(), fg_got.Ow.toarray())
            plot_graph(ax_graph_got, self.ap3.layout(), title='GOT')

            # Fring Graph at convergence
            ax_graph_conv = gather_matrices(
                self.ap3_fg.Iw.toarray(), self.ap3_fg.Dw.toarray(), self.ap3_fg.Ow.toarray()
            )

            plot_graph(ax_graph_conv, self.ap3.layout(), title='Result Test')


def create_imputer(name, sax_in, sax_out, return_dirs=False):

    driver = NumpyDriver()
    tmpdiri, tmpdiro = driver.TempDir(name, suffix='in', create=True), driver.TempDir(name, suffix='out', create=True)

    # Create I/O and save it into tmpdir files
    driver.write_file(sax_in, driver.join(tmpdiri.path, 'forward.npz'), is_sparse=True)
    driver.write_file(sax_out, driver.join(tmpdiri.path, 'backward.npz'), is_sparse=True)

    # Create and init imputer
    imputer = DoubleArrayImputer('test', tmpdiri.path, tmpdiro.path)
    imputer.read_raw_data('forward.npz', 'backward.npz')
    imputer.run_preprocessing()
    imputer.write_features('forward.npz', 'backward.npz')
    imputer.stream_features()

    if return_dirs:
        return imputer, tmpdiri, tmpdiro

    tmpdiri.remove(), tmpdiro.remove()

    return imputer

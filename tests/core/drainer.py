# Global imports
import unittest
from scipy.sparse import csc_matrix

# Local import
from deyep.core.firing_graph.utils import gather_matrices
from deyep.core.imputer.array import DoubleArrayImputer
from deyep.utils.driver.nmp import NumpyDriver
from deyep.core.solver.drainer import FiringGraphDrainer
from deyep.utils.interactive_plots import plot_graph
from deyep.utils.test_pattern import AndPattern2 as ap2#, AndPattern3 as ap3

__maintainer__ = 'Pierre Gouedard'


class TestDrainer(unittest.TestCase):
    def setUp(self):

        # enable, disable visual inspection of graph
        self.visual = False

        # Create And pattern of depth 2
        self.ni, self.no, self.w0 = 10, 2, 10
        self.ap2 = ap2(self.ni, self.no, w=self.w0, seed=1234)
        self.ap2_fg = self.ap2.build_graph_pattern_init()

        # Create And pattern of depth 3
        # self.ni, self.no = 10, 2
        # self.ap3 = ap3(self.ni, self.no)
        # self.ap3_fg = self.ap3.build_graph_pattern_init()

    def andpattern2(self):

            """
            Test And Pattern of depth 2
            python -m unittest tests.core.drainer.TestDrainer.andpattern2

            """

            # Create I/O and save it into tmpdir files
            # TODO /!\/!\/!\/!\/!\ There is a massive problem with streaming of imputer /!\/!\/!\/!\/!\/!\/!\/!\
            ax_input, ax_output = self.ap2.generate_io_sequence(1000, seed=1234)
            imputer = create_imputer('andpattern2', csc_matrix(ax_input), csc_matrix(ax_output))

            # Create drainer
            drainer = FiringGraphDrainer(
                1, self.ap2_fg, imputer, depth=self.ap2.depth, verbose=1, track_backward=True, track_forward=True
            )
            drainer.drain(n=1000)

            # Get Data and assert result is as expected
            model_fg = self.ap2.build_graph_pattern_final()
            track_ibp, track_if, I = drainer.track_ibp, drainer.track_if, drainer.firing_graph.Iw

            self.assertTrue((drainer.firing_graph.I.toarray() == model_fg.I.toarray()).all())
            self.assertTrue(all([I[j, 0] == track_ibp[j, 0] + self.w0 for j in self.ap2.target[0]]))
            self.assertTrue(all([I[j, 1] == track_ibp[j, 1] + self.w0 for j in self.ap2.target[1]]))
            self.assertTrue(all([(-3 < track_ibp[j, 0] - track_if[0, j] < 0)for j in self.ap2.target[0]]))
            self.assertTrue(all([(-3 < track_ibp[j, 1] - track_if[0, j] < 0)for j in self.ap2.target[1]]))

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
        raise NotImplementedError


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

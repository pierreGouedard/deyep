# Global imports
import unittest
import numpy as np
from scipy.sparse import csc_matrix

from deyep.core.firing_graph.utils import mat_from_tuples
from deyep.core.firing_graph.graph import FiringGraph
from deyep.core.imputer.array import DoubleArrayImputer
from deyep.core.solver.drainer import FiringGraphDrainer
from deyep.utils.driver.nmp import NumpyDriver

__maintainer__ = 'Pierre Gouedard'


class TestEquations(unittest.TestCase):
    def setUp(self):

        np.random.seed(392)

        # Create a simple deep network (2 input vertices, 3 network vertices,, 2 output vertices)
        self.n_i, self.n_rn, self.n_o, self.capacity = 2, 4, 2, 10
        l_edges = [('input_0', 'core_0'), ('core_0', 'core_1')] +  \
                  [('input_0', 'core_2'), ('core_2', 'core_1')] + \
                  [('input_1', 'core_3'), ('core_3', 'output_1')]

        # Get matrices from list of edges and build network
        self.sax_I, self.sax_D, self.sax_O = mat_from_tuples(l_edges, self.n_i, self.n_rn, self.n_o)
        self.fg = FiringGraph.from_matrices(
            'test_equation', self.sax_D, self.sax_I, self.sax_O, self.capacity, 100, [1, 1, 1, 1]
        )
        self.input = csc_matrix(([1., 1.], ([0, 1], [1, 0])), shape=(10, 2), dtype=int)
        self.output = csc_matrix(([1., 1.], ([0, 2], [1, 0])), shape=(10, 2), dtype=int)

    def forward(self):
        """
        Very precise case on very simple graph to validate basics of drainer for forward equations
        python -m unittest tests.core.equations.TestEquations.forward

        """

        imputer = init_imputer(self.input, self.output)
        drainer = FiringGraphDrainer(1, self.fg, imputer, 2)

        # Run for one epoch (forward transmitting)
        drainer.drain(1)
        self.assertAlmostEqual(len(drainer.sax_so.nonzero()[0]), 0)
        self.assertAlmostEqual(len(np.unique(drainer.sax_sn.nonzero()[1])), 1)
        self.assertAlmostEqual(np.unique(drainer.sax_sn.nonzero()[1])[0], 3)

        # Run for one other epoch (forward processing)
        drainer.drain(1)
        vertex = drainer.firing_graph.core_vertices[3]
        self.assertEqual(len([t for t in vertex.basis.queue_forward if t is not None]), 1)
        self.assertEqual(vertex.basis.queue_forward[0], (0, ['k=0,N=21']))

        for vertex in drainer.firing_graph.core_vertices[:-1]:
            self.assertEqual(len([t for t in vertex.basis.queue_forward if t is not None]), 0)

        # Run epoch (2nd forward transmitting)
        drainer.drain(1)
        self.assertAlmostEqual(len(np.unique(drainer.sax_so.nonzero()[1])), 1)
        self.assertAlmostEqual(np.unique(drainer.sax_so.nonzero()[1])[0], 1)
        self.assertAlmostEqual(len(np.unique(drainer.sax_sn.nonzero()[1])), 2)

        # Run one epoch (backward buffer and 2nd forward processing)
        drainer.drain(1)
        d_test = {0: 1, 1: 0, 2: 1, 3: 1}
        for i, v in d_test.items():
            vertex = drainer.firing_graph.core_vertices[i]
            self.assertEqual(len([t for t in vertex.basis.queue_forward if t is not None]), v)

        self.assertTrue((drainer.sax_sab.toarray() == np.array([[False, False, False, True]] * 2, dtype=bool)).all())
        self.assertTrue((drainer.sax_sab.toarray() == np.array([[False, False, False, True]] * 2, dtype=bool)).all())
        self.assertEqual(np.unique(drainer.sax_sob.nonzero()[1])[0], 1)

    def backward(self):
        """
        Very precise case on very simple graph to validate basics of drainer for ackward equations
        python -m unittest tests.core.equations.TestEquations.backward

        """

        imputer = init_imputer(self.input, self.output)
        drainer = FiringGraphDrainer(1, self.fg, imputer, 2)

        # Run for sufficient number of epoch to reach first backward processing
        ax_out = drainer.firing_graph.O.toarray()
        drainer.drain(5)

        vertex = drainer.firing_graph.core_vertices[-1]
        self.assertTrue((ax_out == drainer.firing_graph.O.toarray()).all())

        ax_out = drainer.sax_sob.toarray().transpose()
        ax_test = vertex.basis.base.toarray()[0]
        self.assertTrue((ax_out.dot(ax_test) == np.array([0, 1])).all())

        ax_test = vertex.basis.base_from_key('k={},N={}'.format(vertex.basis.key, vertex.basis.N), offset=1).toarray()[0]
        self.assertTrue((ax_out.dot(ax_test) == np.array([0, 1])).all())
        self.assertTrue((drainer.sax_snb.toarray() == 0).all())

        # Run first backward transmitting
        ax_Ow, ax_Dw, ax_Iw = drainer.firing_graph.Ow, drainer.firing_graph.Dw, drainer.firing_graph.Iw
        drainer.drain(1)

        # Make sure update of network is ok
        self.assertEqual(ax_Ow[-1, 1], drainer.firing_graph.Ow[-1, 1] - 1)
        self.assertEqual(len(drainer.firing_graph.Ow.nonzero()[0]), 1)
        self.assertTrue(not (ax_Dw != drainer.firing_graph.Dw).toarray().any())
        self.assertTrue(not (ax_Iw != drainer.firing_graph.Iw).toarray().any())

        # Make sure vertices have received their backward message
        self.assertEqual(len(np.unique(drainer.sax_snb.nonzero()[1])), 1)

        # Run second backward processing
        drainer.drain(1)

        # Make sure no candidate has been updated
        self.assertEqual(len(drainer.firing_graph.Ow.nonzero()[0]), 1)

        # Make sure sax_sob is all zero
        self.assertEqual(drainer.sax_sob.nnz, 0)

        # make sure backward messages are correct
        vertex = drainer.firing_graph.input_vertices[-1]

        self.assertEqual(drainer.sax_snb[:, -1].toarray()[:, 0].dot(vertex.basis.base.toarray()[0]), 1)
        self.assertEqual(vertex.basis.depth_from_basis(drainer.sax_snb[:, -1].transpose()), ([0], [1.]))

        # Backward transmiting
        Iw = drainer.firing_graph.Iw
        drainer.drain(1)

        # Assert Iw has been updated
        self.assertEqual(drainer.firing_graph.Iw[1, -1], Iw[1, -1] + 1)

        # just to make sure that the rest is ok
        drainer.drain(10)


def init_imputer(ax_input, ax_output):
    # Create temporary directory for test
    driver = NumpyDriver()
    tmpdirin, tmpdirout = driver.TempDir('test_equation', suffix='in', create=True), \
                          driver.TempDir('test_equation', suffix='out', create=True)

    # Create I/O and save it into tmpdir files
    driver.write_file(ax_input, driver.join(tmpdirin.path, 'forward.npz'), is_sparse=True)
    driver.write_file(ax_output, driver.join(tmpdirin.path, 'backward.npz'), is_sparse=True)

    # Create and init imputer
    imputer = DoubleArrayImputer('test', tmpdirin.path, tmpdirout.path)
    imputer.read_raw_data('forward.npz', 'backward.npz')
    imputer.run_preprocessing()
    imputer.write_features('forward.npz', 'backward.npz')
    imputer.stream_features()

    tmpdirin.remove()
    tmpdirout.remove()

    return imputer
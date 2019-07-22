# Global imports
import unittest
import numpy as np

# Local import
from deyep.core.firing_graph.utils import set_vertices_from_mat, set_frequencies, mat_from_tuples
from deyep.core.firing_graph.graph import FiringGraph

__maintainer__ = 'Pierre Gouedard'


class TestBuilder(unittest.TestCase):
    def setUp(self):

        # Create a simple deep network (2 input nodes, 3 network nodes,, 2 output nodes)
        n_i, n_rn, n_o, self.capacity = 2, 5, 3, 20
        l_edges = [('input_0', 'core_0'), ('core_0', 'core_1'), ('core_0', 'core_2'),
                   ('core_1', 'output_0'), ('core_2', 'output_1')] +  \
                  [('input_0', 'core_3'), ('core_3', 'core_2')] + \
                  [('input_1', 'core_4'), ('core_4', 'output_2')] + \
                  [('input_1', 'core_2')]

        # Get matrices for building test
        self.sax_in, self.sax_core, self.sax_out = mat_from_tuples(l_edges, n_i, n_rn, n_o)

    def spread_frequency(self):
        """
        Test basic graph building and frequency distribution
        python -m unittest tests.core.builder.TestBuilder.spread_frequency

        """

        # Build dictionary of nodes
        d_inputs = set_vertices_from_mat(self.sax_in, 'input')
        d_outputs = set_vertices_from_mat(self.sax_out, 'output')
        d_cores = set_vertices_from_mat(self.sax_core, 'core')

        self.assertEqual(len(d_inputs), self.sax_in.shape[0])
        self.assertEqual(len(d_cores), self.sax_core.shape[0])
        self.assertEqual(len(d_outputs), self.sax_out.shape[-1])

        # distribute frequency among input nodes,
        d_inputs, set_freqs, d_forward_freqs = set_frequencies(d_inputs, {0}, 1, {})

        # distribute frequencies among network nodes
        d_cores, set_freqs_, d_forward_freqs = set_frequencies(
            d_cores, {0}, self.capacity, d_forward_freqs, offset=len(set_freqs)
        )

        # Assert that no frequency are conflicting
        self.assertEqual(set_freqs_, {0, 1})
        self.assertEqual(set_freqs, {0})
        self.assertTrue(all([d['freqs'] == [0] for _, d in d_inputs.items()]))
        self.assertEqual(d_cores[3]['freqs'], [self.capacity + 1])
        self.assertEqual(d_cores[0]['freqs'], [1])

        d_got = {'core_0': [0], 'core_1': [1], 'core_2': [0, 1, 21], 'core_3': [0], 'core_4': [0]}
        self.assertEqual(d_forward_freqs, d_got)
        self.assertEqual(((max(set_freqs_) + 1) * self.capacity) + len(set_freqs), (self.capacity * 2) + 1)

    def building_graph(self):
        """
        Test basic graph building and frequency distribution
        python -m unittest tests.core.builder.TestBuilder.building_graph

        """
        # Create mask drainer and create firing graph
        mask_drain = {'I': np.zeros(self.sax_in.shape[0]), 'D': np.ones(self.sax_core.shape[0])}
        firing_graph = FiringGraph.from_matrices(
            'test_basis', self.sax_core, self.sax_in, self.sax_out, self.capacity,  100, [1, 1, 1, 1, 1], mask_drain
        )

        # Assert matrice are correct
        self.assertTrue((firing_graph.D_mask.toarray() == (self.sax_core > 0).toarray()).all())
        self.assertTrue((firing_graph.I_mask.toarray() == np.zeros(self.sax_in.shape)).all())
        self.assertTrue((firing_graph.Iw.toarray() == self.sax_in.toarray()).all())
        self.assertTrue((firing_graph.Ow.toarray() == self.sax_out.toarray()).all())

        # Create mask drainer and create firing graph
        mask_drain = {'I': np.ones(self.sax_in.shape[0]), 'D': np.zeros(self.sax_core.shape[0])}
        firing_graph = FiringGraph.from_matrices(
            'test_basis', self.sax_core, self.sax_in, self.sax_out, self.capacity, 100, [1, 1, 1, 1, 1], mask_drain
        )

        # Assert matrices are correct
        self.assertTrue((firing_graph.D_mask.toarray() == np.zeros(self.sax_core.shape)).all())
        self.assertTrue((firing_graph.I_mask.toarray() == (self.sax_in > 0).toarray()).all())



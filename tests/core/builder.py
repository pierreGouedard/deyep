# Global imports
import unittest
from scipy.sparse import csc_matrix as csc

# Local import
from deyep.core.firing_graph.utils import set_vertices_from_mat, set_frequencies, mat_from_tuples


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
        self.mat_in, self.mat_net, self.mat_out = mat_from_tuples(l_edges, n_i, n_rn, n_o)

        # Get matrices for refining test
        self.sax_D, self.sax_I, self.sax_O, self.sax_Cm = csc((5, 5)), csc((1, 5)), csc((5, 2)), csc((5, 2))
        self.sax_I[0, [0, 1]], self.sax_D[0, 2], self.sax_D[1, 3], self.sax_D[2, 4] = 1, 1, 1, 1

    def building(self):
        """
        Test basic graph building and frequency distribution
        python -m unittest tests.core.builder.TestBuilder.building

        """

        # Build dictionary of nodes
        d_inputs = set_vertices_from_mat(self.mat_in, 'input')
        d_outputs = set_vertices_from_mat(self.mat_out, 'output')
        d_cores = set_vertices_from_mat(self.mat_net, 'core')

        self.assertEqual(len(d_inputs), self.mat_in.shape[0])
        self.assertEqual(len(d_cores), self.mat_net.shape[0])
        self.assertEqual(len(d_outputs), self.mat_out.shape[-1])

        # distribute frequency among input nodes,
        d_inputs, set_freqs, d_forward_freqs = set_frequencies(d_inputs, {0}, 1, {})

        # distribute frequencies among network nodes
        d_networks, set_freqs_, d_forward_freqs = set_frequencies(
            d_cores, {0}, self.capacity, d_forward_freqs, offset=len(set_freqs)
        )

        # Assert that no frequency are conflicting
        self.assertEqual(set_freqs_, {0, 1})
        self.assertEqual(set_freqs, {0})
        self.assertTrue(all([d['freqs'] == [0] for _, d in d_inputs.items()]))
        self.assertEqual(d_networks[3]['freqs'], [self.capacity + 1])
        self.assertEqual(d_networks[0]['freqs'], [1])

        d_got = {'network_0': [0], 'network_1': [1], 'network_2': [0, 1, 21], 'network_3': [0], 'network_4': [0]}
        self.assertEqual(d_forward_freqs, d_got)

        self.assertEqual(((max(set_freqs_) + 1) * self.capacity) + len(set_freqs), (self.capacity * 2) + 1)

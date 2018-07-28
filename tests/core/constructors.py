# Global imports
import unittest

# Local import
from deyep.core.builder.comon import set_nodes_from_mat, set_frequencies, mat_from_tuples

__maintainer__ = 'Pierre Gouedard'


class TestConstructor(unittest.TestCase):
    def setUp(self):

        # Create a simple deep network (2 input nodes, 3 network nodes,, 2 output nodes)
        n_i, n_rn, n_o, self.capacity = 2, 5, 3, 20
        l_edges = [('input_0', 'network_0'), ('network_0', 'network_1'), ('network_0', 'network_2'),
                   ('network_1', 'output_0'), ('network_2', 'output_1')] +  \
                  [('input_0', 'network_3'), ('network_3', 'network_2')] + \
                  [('input_1', 'network_4'), ('network_4', 'output_2')] + \
                  [('input_1', 'network_2')]

        # Get matrices from list of edges and build network
        self.mat_in, self.mat_net, self.mat_out = mat_from_tuples(l_edges, n_i, n_rn, n_o)

    def test_deepnet_building(self):
        """
        Test basic frequency management
        python -m unittest tests.core.constructors.TestConstructor.test_deepnet_building

        """

        # Build dictionary of nodes
        d_inputs = set_nodes_from_mat(self.mat_in, 'input')
        d_outputs = set_nodes_from_mat(self.mat_out, 'output')
        d_networks = set_nodes_from_mat(self.mat_net, 'network')

        self.assertEqual(len(d_inputs), self.mat_in.shape[0])
        self.assertEqual(len(d_networks), self.mat_net.shape[0])
        self.assertEqual(len(d_outputs), self.mat_out.shape[-1])

        # distribute frequency among input nodes,
        d_inputs, set_freqs, d_forward_freqs = set_frequencies(d_inputs, {0}, 1, {})

        # distribute frequencies among network nodes
        d_networks, set_freqs_, d_forward_freqs = set_frequencies(d_networks, {0}, self.capacity, d_forward_freqs,
                                                                  offset=len(set_freqs))

        # Assert that no frequency are conflicting
        self.assertEqual(set_freqs_, {0, 1})
        self.assertEqual(set_freqs, {0})
        self.assertTrue(all([d['freqs'] == [0] for _, d in d_inputs.items()]))
        self.assertEqual(d_networks[3]['freqs'], [self.capacity + 1])
        self.assertEqual(d_networks[0]['freqs'], [1])
        self.assertEqual(d_forward_freqs, {'network_0': {0}, 'network_1': {1}, 'network_2': {0, 1, 21},
                                           'network_3': {0}, 'network_4': {0}})
        self.assertEqual(((max(set_freqs_) + 1) * self.capacity) + len(set_freqs), (self.capacity * 2) + 1 )

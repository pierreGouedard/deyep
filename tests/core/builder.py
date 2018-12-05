# Global imports
import unittest
from scipy.sparse import csc_matrix as csc
import numpy as np

# Local import
from deyep.core.builder.comon import set_nodes_from_mat, set_frequencies, mat_from_tuples
from deyep.core.builder.binomial import BinomialGraphBuilder


__maintainer__ = 'Pierre Gouedard'


class TestBuilder(unittest.TestCase):
    def setUp(self):

        # Create a simple deep network (2 input nodes, 3 network nodes,, 2 output nodes)
        n_i, n_rn, n_o, self.capacity = 2, 5, 3, 20
        l_edges = [('input_0', 'network_0'), ('network_0', 'network_1'), ('network_0', 'network_2'),
                   ('network_1', 'output_0'), ('network_2', 'output_1')] +  \
                  [('input_0', 'network_3'), ('network_3', 'network_2')] + \
                  [('input_1', 'network_4'), ('network_4', 'output_2')] + \
                  [('input_1', 'network_2')]

        # Get matrices for building test
        self.mat_in, self.mat_net, self.mat_out = mat_from_tuples(l_edges, n_i, n_rn, n_o)

        # Get matrices for refining test
        self.sax_D, self.sax_I, self.sax_O, self.sax_Cm = csc((5, 5)), csc((1, 5)), csc((5, 2)), csc((5, 2))
        self.sax_I[0, [0, 1]], self.sax_D[0, 2], self.sax_D[1, 3], self.sax_D[2, 4] = 1, 1, 1, 1

    def building(self):
        """
        Test basic frequency management
        python -m unittest tests.core.builder.TestBuilder.building

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
        self.assertEqual(d_forward_freqs, {'network_0': [0], 'network_1': [1], 'network_2': [0, 1, 21],
                                           'network_3': [0], 'network_4': [0]})
        self.assertEqual(((max(set_freqs_) + 1) * self.capacity) + len(set_freqs), (self.capacity * 2) + 1)

    def refining(self):
        """
        python -m unittest tests.core.builder.TestBuilder.refining

        """
        # Case 1 (delay)
        sax_Cm = self.sax_Cm.copy().astype(bool)
        sax_Cm[:2, :] = True

        d_depth = BinomialGraphBuilder.compute_depth(
            self.sax_D.copy().transpose(), (sax_Cm < 1).transpose()
        )

        sax_I, sax_D = BinomialGraphBuilder.update_structure(self.sax_I.copy(), self.sax_D.copy(), d_depth)

        self.assertTrue((sax_I.astype(int).toarray() == np.array([2, 1, 0, 0, 0])).all())
        self.assertTrue((sax_D[0, :].astype(int).toarray() == np.array([0, 0, 2, 0, 0])).all())
        self.assertTrue((sax_D[1, :].astype(int).toarray() == np.array([0, 0, 0, 1, 0])).all())
        self.assertTrue((sax_D[2, :].astype(int).toarray() == np.array([0, 0, 0, 0, 1])).all())
        self.assertEqual(sax_D[3:, :].sum(), 0)

        # Case 2 (no delay)
        sax_Cm = self.sax_Cm.copy().astype(bool)

        d_depth = BinomialGraphBuilder.compute_depth(
            self.sax_D.copy().transpose(), (sax_Cm < 1).transpose()
        )

        sax_I, sax_D = BinomialGraphBuilder.update_structure(self.sax_I.copy(), self.sax_D.copy(), d_depth)

        self.assertTrue((sax_I.astype(int).toarray() == np.array([3, 2, 0, 0, 0])).all())
        self.assertTrue((sax_D[0, :].astype(int).toarray() == np.array([0, 0, 2, 0, 0])).all())
        self.assertTrue((sax_D[1, :].astype(int).toarray() == np.array([0, 0, 0, 1, 0])).all())
        self.assertTrue((sax_D[2, :].astype(int).toarray() == np.array([0, 0, 0, 0, 1])).all())
        self.assertEqual(sax_D[3:, :].sum(), 0)


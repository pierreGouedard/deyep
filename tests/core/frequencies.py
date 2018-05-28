# Global imports
import numpy as np
import unittest
from deyep.utils.names import KVName

# Local import
from tests.comon import get_mat_from_path
from deyep.core.constructors.constructors import Constructor
from deyep.core.tools.linear_algebra import inner_product, get_fourrier_series as series

__maintainer__ = 'Pierre Gouedard'


class TestFrequencyStack(unittest.TestCase):
    def setUp(self):

        # Create a simple deep network (2 input nodes, 3 network nodes,, 2 output nodes)
        self.n_i, self.n_rn, self.n_o, self.capacity = 2, 5, 3, 10
        l_edges = [('input_0', 'network_0'), ('network_0', 'network_1'), ('network_0', 'network_2'),
                   ('network_1', 'output_0'), ('network_2', 'output_1')] +  \
                  [('input_0', 'network_3'), ('network_3', 'network_2')] + \
                  [('input_1', 'network_4'), ('network_4', 'output_2')] + \
                  [('input_1', 'network_2')]

        # Get matrices from list of edges and build network
        mat_in, mat_net, mat_out = get_mat_from_path(l_edges, self.n_i, self.n_rn, self.n_o)

        self.deep_network_1 = Constructor.from_weighted_direct_matrices(mat_net, mat_in, mat_out, self.capacity)

    def test_frequency_stack_basics(self):
        """
        Test basic frequency management
        python -m unittest tests.core.frequencies.TestFrequencyStack.test_frequency_stack_basics

        """
        N = self.deep_network_1.input_nodes[0].frequency_stack.N

        # Assert correct initialization of stacks
        for i in range(self.n_i):
            str_ = KVName.from_dict({'N': str(N), 'k': 0}).to_string()
            self.assertTrue(str_ in self.deep_network_1.input_nodes[i].frequency_stack.setfree)

        for i in range(self.n_rn):
            self.assertEqual(len(self.deep_network_1.network_nodes[i].frequency_stack.setfree), self.capacity)

        stack = self.deep_network_1.network_nodes[0].frequency_stack
        assert ((len(stack.setfree) == self.capacity) & (len(stack.map) == 0) & (len(stack.priorities) == 0))

        # Test single encoding
        s_in = self.deep_network_1.network_nodes[1].frequency_stack.fourrier_basis()[0, :]
        coef_in = stack.coef_from_series(s_in, basis=[stack.coef_from_key(k) for k in stack.setfree])
        s_out = stack.encode(s_in)
        coef_out = stack.coef_from_series(s_out, basis=[stack.coef_from_key(k) for k in stack.map.keys()])

        self.assertTrue(stack.key_from_coef(coef_out) not in stack.setfree)
        self.assertTrue(stack.key_from_coef(coef_out) in stack.map.keys())
        self.assertTrue(stack.key_from_coef(coef_in) in stack.map[stack.key_from_coef(coef_out)])

        l_ = [inner_product(s_out, s_b) for s_b in stack.fourrier_basis()]
        self.assertAlmostEqual(np.real(sum(l_)), 0., delta=1e-10)
        l_ = [inner_product(s_out, s_b) for s_b in stack.fourrier_basis(free=False)]
        self.assertAlmostEqual(np.real(sum(l_)), 1., delta=1e-10)

        # Test decoding
        s_in_ = stack.decode(s_out)
        self.assertAlmostEqual((s_in_ - s_in).sum(), 0., delta=1e-10)

        stack.release_key(p=1)
        self.assertTrue(stack.key_from_coef(coef_out) in stack.setfree)

    def test_frequency_stack_advanced(self):
        """
        Test advanced frequency management (encoding and so)
        python -m unittest tests.core.frequencies.TestFrequencyStack.test_frequency_stack_advanced

        """
        stack = self.deep_network_1.network_nodes[0].frequency_stack
        l_coefs, coef = [], self.deep_network_1.network_nodes[1].frequency_stack.fourrier_basis()[0]
        i = 0
        for i in range(self.capacity - 1):
            l_coefs += [stack.key_from_coef(stack.encode([coef]))]

        self.assertEqual(stack.priorities[l_coefs[-1]], i)
        self.assertEqual(len(stack.setfree),  1)

        _ = stack.key_from_coef(stack.encode([coef]))

        self.assertEqual(len(stack.setfree), int(0.3 * self.capacity))
        self.assertTrue(all([k in stack.setfree for k in l_coefs[:int(0.3 * self.capacity)]]))

    def test_frequency_orthogonality(self):
        """
        Test that may appear in another test set (like algebra)
        python -m unittest tests.core.frequencies.TestFrequencyStack.test_frequency_orthogonality

        """

        stack_x = self.deep_network_1.network_nodes[0].frequency_stack
        stack_y = self.deep_network_1.network_nodes[3].frequency_stack

        # Get Fourrier basis coef for network node 0 and 3
        coef_basis_x = stack_x.fourrier_basis()
        coef_basis_y = stack_y.fourrier_basis()

        self.assertEqual(set(coef_basis_x).intersection(set(coef_basis_y)), set())

        # Get the corresponding Fourrier series
        series_x = sum([series(c) for c in coef_basis_x])
        series_y = sum([series(c) for c in coef_basis_y])

        res = inner_product(series_x, series_y)

        self.assertAlmostEqual(np.real(res), 0, delta=1e-10)

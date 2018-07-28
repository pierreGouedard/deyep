# Global imports
import numpy as np
import unittest
from deyep.utils.names import KVName

# Local import
from deyep.core.tools.linear_algebra.fourrier_domain import inner_product
from deyep.core.builder.comon import mat_from_tuples
from deyep.core.deep_network import DeepNetwork

__maintainer__ = 'Pierre Gouedard'


class TestFourrierBasis(unittest.TestCase):
    def setUp(self):

        # Create a simple deep network (2 input nodes, 3 network nodes,, 2 output nodes)
        self.n_i, self.n_rn, self.n_o, self.capacity = 2, 5, 3, 10
        l_edges = [('input_0', 'network_0'), ('network_0', 'network_1'), ('network_0', 'network_2'),
                   ('network_1', 'output_0'), ('network_2', 'output_1')] +  \
                  [('input_0', 'network_3'), ('network_3', 'network_2')] + \
                  [('input_1', 'network_4'), ('network_4', 'output_2')] + \
                  [('input_1', 'network_2')]

        # Get matrices from list of edges and build network
        mat_in, mat_net, mat_out = mat_from_tuples(l_edges, self.n_i, self.n_rn, self.n_o)

        self.deep_network_1 = DeepNetwork.from_matrices(mat_net, mat_in, mat_out, self.capacity)

    def test_fourrier_basics(self):
        """
        Test basic frequency management
        python -m unittest tests.core.fourrier_basis.TestFourrierBasis.test_fourrier_basics

        """
        Basis = self.deep_network_1.network_nodes[2].basis
        self.assertEqual(len(Basis.basis), self.capacity)

        # Make sure that nodes with edges toward same nodes has orthogonal base
        s_net1, s_net2 = self.deep_network_1.network_nodes[3].basis.base, self.deep_network_1.network_nodes[1].basis.base
        self.assertEqual(np.round(np.real(inner_product(s_net1,  s_net2))), 0.)

        # Test single encoding
        s_in = self.deep_network_1.network_nodes[1].basis.base
        keys_in = Basis.keys_from_forward_basis(s_in)
        s_out = Basis.encode(s_in, 0)

        self.assertEqual(keys_in, ['N=21,k=1'])
        self.assertEqual(np.round(np.real(inner_product(s_out,  Basis.base))), 1.)
        self.assertEqual(len(filter(lambda x: x is not None, Basis.queue_forward)), 1)

        # Test double encoding
        s_in += self.deep_network_1.network_nodes[3].basis.base
        keys_in = Basis.keys_from_forward_basis(s_in)
        s_out = Basis.encode(s_in, 1)
        import IPython
        IPython.embed()
        self.assertEqual(set(keys_in), {'N=21,k=1', 'N=21,k=11'})
        self.assertEqual(np.round(np.real(inner_product(s_out,  Basis.base))), 1.)
        w


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
        l_keys, s_in = [], self.deep_network_1.network_nodes[1].frequency_stack.fourrier_basis()[0]
        i = 0
        for i in range(self.capacity - 1):
            l_keys += [stack.key_from_coef(stack.coef_from_series(stack.encode(s_in), stack.basis_specific))]

        self.assertEqual(stack.priorities[l_keys[-1]], i)
        self.assertEqual(len(stack.setfree),  1)

        _ = stack.key_from_coef(stack.coef_from_series(stack.encode(s_in), stack.basis_specific))

        self.assertEqual(len(stack.setfree), int(0.3 * self.capacity))
        self.assertTrue(all([k in stack.setfree for k in l_keys[:int(0.3 * self.capacity)]]))

    def test_frequency_orthogonality(self):
        """
        Test that may appear in another test set (like algebra)
        python -m unittest tests.core.frequencies.TestFrequencyStack.test_frequency_orthogonality

        """

        stack_x = self.deep_network_1.network_nodes[0].frequency_stack
        stack_y = self.deep_network_1.network_nodes[3].frequency_stack

        # Get Fourrier basis coef for network node 0 and 3
        ax_basis_x = stack_x.fourrier_basis().sum(axis=0)
        ax_basis_y = stack_y.fourrier_basis().sum(axis=0)

        res = inner_product(ax_basis_x, ax_basis_y)

        self.assertAlmostEqual(np.real(res), 0, delta=1e-10)

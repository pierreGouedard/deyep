# Global imports
import numpy as np
import unittest

# Local import
from deyep.core.tools.linear_algebra.natural_domain import inner_product
from deyep.core.builder.comon import mat_from_tuples
from deyep.core.deep_network import DeepNetwork

__maintainer__ = 'Pierre Gouedard'


class TestCanonicalBasis(unittest.TestCase):
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

        self.deep_network_1 = DeepNetwork.from_matrices('test', mat_net, mat_in, mat_out, self.capacity, 'canonical')

    def test_canonical_basics(self):
        """
        Test basic frequency management
        python -m unittest tests.core.canonical_basis.TestCanonicalBasis.test_canonical_basics

        """

        Basis = self.deep_network_1.network_nodes[2].basis
        self.assertEqual(len(Basis.basis), self.capacity)

        # Make sure that nodes with edges toward same nodes has orthogonal base
        s_net1, s_net2 = self.deep_network_1.network_nodes[3].basis.base, self.deep_network_1.network_nodes[1].basis.base
        self.assertEqual(inner_product(s_net1,  s_net2), 0.)

        # Test single encoding
        s_in1 = self.deep_network_1.network_nodes[1].basis.base
        keys_in = Basis.keys_from_forward_basis(s_in1)
        s_out = Basis.encode(s_in1, 0)

        self.assertEqual(keys_in, ['N=21,k=1'])
        self.assertEqual(np.round(np.real(inner_product(s_out,  Basis.base))), 1.)
        self.assertEqual(len(filter(lambda x: x is not None, Basis.queue_forward)), 1)

        # Test double encoding
        s_in2 = s_in1 + self.deep_network_1.network_nodes[3].basis.base
        keys_in = Basis.keys_from_forward_basis(s_in2)
        s_out = Basis.encode(s_in2, 1)

        self.assertEqual(set(keys_in), {'N=21,k=1', 'N=21,k=11'})
        self.assertEqual(np.round(np.real(inner_product(s_out,  Basis.base))), 1.)
        self.assertEqual(len(filter(lambda x: x is not None, Basis.queue_forward)), 2)

        s_out_ = s_out + Basis.base_from_key('k={},N={}'.format(int(Basis.key), Basis.N), offset=1)

        # Test decoding single encoding
        s_in1_ = Basis.decode(s_out_, t=2)
        self.assertEqual(np.round(np.real(inner_product(s_in1_, s_in1))), 1.)

        # Test decoding double encoding
        s_in2_ = Basis.decode(s_out_, t=3)
        self.assertEqual(np.round(np.real(inner_product(s_in2_, s_in2))), 2.)

    def test_canonical_limits(self):
        """
        Test advanced frequency management (encoding and so)
        python -m unittest tests.core.fourrier_basis.TestFourrierBasis.test_fourrier_limits

        """
        # TODO when the queue is full, does it have the correct behaviour ?
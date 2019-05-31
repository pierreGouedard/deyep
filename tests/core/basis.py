# Global imports
import unittest

import numpy as np

from deyep.core.firing_graph.utils import mat_from_tuples
from deyep.core.firing_graph.graph import FiringGraph
from deyep.core.tools.linear_algebra.comon import inner_product

__maintainer__ = 'Pierre Gouedard'


class TestCanonicalBasis(unittest.TestCase):
    def setUp(self):

        # Create a simple deep network (2 input nodes, 3 network nodes,, 2 output nodes)
        self.n_i, self.n_rn, self.n_o, self.capacity = 2, 5, 3, 10
        l_edges = [('input_0', 'core_0'), ('core_0', 'core_1'), ('core_0', 'core_2'),
                   ('core_1', 'output_0'), ('core_2', 'output_1')] +  \
                  [('input_0', 'core_3'), ('core_3', 'core_2')] + \
                  [('input_1', 'core_4'), ('core_4', 'output_2')] + \
                  [('input_1', 'core_2')]

        # Get matrices from list of edges and build network
        mat_in, mat_net, mat_out = mat_from_tuples(l_edges, self.n_i, self.n_rn, self.n_o)

        self.firing_graph = FiringGraph.from_matrices(
            'test_basis', mat_net, mat_in, mat_out, self.capacity,  100, [1, 1, 1, 1, 1])

    def basics(self):
        """
        Test basic frequency management
        python -m unittest tests.core.basis.TestCanonicalBasis.basics

        """

        Basis = self.firing_graph.core_vertices[2].basis
        self.assertEqual(len(Basis.basis), int(self.capacity / 2))

        # Make sure that nodes with edges toward same nodes has orthogonal base
        s_net1, s_net2 = self.firing_graph.core_vertices[3].basis.base, self.firing_graph.core_vertices[1].basis.base
        self.assertEqual(inner_product(s_net1,  s_net2), 0.)

        # Test single encoding
        s_in1 = self.firing_graph.core_vertices[1].basis.base
        keys_in = Basis.keys_from_forward_basis(s_in1)
        s_out = Basis.encode(s_in1, 0, 1)

        self.assertEqual(keys_in, ['N=21,k=1'])
        self.assertEqual(np.round(np.real(inner_product(s_out,  Basis.base))), 1.)
        self.assertEqual(len(filter(lambda x: x is not None, Basis.queue_forward)), 1)

        # Test double encoding
        s_in2 = s_in1 + self.firing_graph.core_vertices[3].basis.base
        keys_in = Basis.keys_from_forward_basis(s_in2)
        s_out = Basis.encode(s_in2, 1, 1)

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

    def advanced(self):
        """
        Test advanced frequency management (encoding and so) refer to
        -m unittest tests.core.fourrier_basis.TestFourrierBasis.test_fourrier_limits
        """
        # TODO
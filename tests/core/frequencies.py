# Global imports
import numpy as np
import unittest
from scipy.sparse import lil_matrix
from deyep.utils.names import KVName

# Local import
from deyep.core.constructors.constructors import Constructor
from deyep.core.tools.linear_algebra import inner_product, get_fourrier_series as series

__maintainer__ = 'Pierre Gouedard'


class TestFrequencyStack(unittest.TestCase):
    def setUp(self):

        # Create a simple deep network (2 input nodes, 3 network nodes,, 2 output nodes)
        self.n_i, self.n_rn, self.n_o, self.capacity = 2, 3, 2, 10
        l_edges = [('input_0', 'network_0'), ('network_0', 'network_1'), ('network_1', 'output_0')] + \
                  [('input_1', 'network_2'), ('network_2', 'output_1')]

        # Get matrices from list of edges and build network
        mat_in, mat_net, mat_out = get_mat_from_path(l_edges, self.n_i, self.n_rn, self.n_o)

        self.deep_network_1 = Constructor.from_weighted_direct_matrices(mat_net, mat_in, mat_out, self.capacity)

    def test_frequency_stack_basics(self):
        """
        Test basic frequency management
        python -m unittest tests.core.frequencies.TestFrequencyStack.test_frequency_stack_basics

        """
        N = (self.n_i + self.n_rn) * self.capacity

        # Assert correct initialization of stacks
        for i in range(self.n_i):
            for c in range(self.capacity):
                str_ = KVName.from_dict({'N': str(N), 'k': str((i * self.capacity) + c)}).to_string()
                self.assertTrue(str_ in self.deep_network_1.input_nodes[i].frequency_stack.setfree)

        for i in range(self.n_rn):
            for c in range(self.capacity):
                str_ = KVName.from_dict({'N': str(N), 'k': str(((i + 2) * self.capacity) + c)}).to_string()
                self.assertTrue(str_ in self.deep_network_1.network_nodes[i].frequency_stack.setfree)

        stack = self.deep_network_1.network_nodes[0].frequency_stack
        assert ((len(stack.setfree) == self.capacity) & (len(stack.map) == 0) & (len(stack.priorities) == 0))

        # Test single encoding
        coef_in = self.deep_network_1.network_nodes[1].frequency_stack.basis[0]
        coef_out = stack.encode(coef_in)

        self.assertTrue(stack.str_from_coef(coef_out) not in stack.setfree)
        self.assertAlmostEqual(np.real(sum([inner_product(series(coef_out), series(b)) for b in stack.basis])), 1., delta=1e-10)

        # Test decoding
        coef_in_ = stack.decode(coef_out)
        self.assertEqual(coef_in, coef_in_)

        stack.release_key(p=1)
        self.assertTrue(stack.str_from_coef(coef_out) in stack.setfree)

    def test_frequency_stack_advanced(self):
        """
        Test advanced frequency management (encoding and so)
        python -m unittest tests.core.frequencies.TestFrequencyStack.test_frequency_stack_advanced

        """

        stack = self.deep_network_1.network_nodes[0].frequency_stack
        l_coefs, coef = [], self.deep_network_1.network_nodes[1].frequency_stack.basis[0]
        i = 0
        for i in range(self.capacity - 1):
            l_coefs += [stack.str_from_coef(stack.encode(coef))]

        self.assertEqual(stack.priorities[l_coefs[-1]], i)
        self.assertEqual(len(stack.setfree),  1)

        _ = stack.str_from_coef(stack.encode(coef))

        self.assertEqual(len(stack.setfree), int(0.3 * self.capacity))
        self.assertTrue(all([k in stack.setfree for k in l_coefs[: int(0.3 * self.capacity)]]))

    def test_frequency_orthogonality(self):
        """
        Test that may appear in another test set (like algebra)
        python -m unittest tests.communication.TestFrequenciesStack.test_

        """

        import IPython
        IPython.embed()

        n = len(self.deep_network_1.input_nodes) + len(self.deep_network_1.network_nodes)
        capacity = self.deep_network_1.network_nodes[0].frequency_stack.capacity

        # Make sure that each input nodes can decode each of their message
        mat_basis = np.zeros([n, capacity])
        for node in self.deep_network_1.input_nodes + self.deep_network_1.network_nodes:
            mat_basis[node.i, :] = node.frequency_stack.basis

            # Implement the complex algebra operation described in scientific report (with execution time test)


def get_mat_from_path(l_edges, n_i, n_rn, n_o, weights='random'):
    # Init matrices
    mat_in = lil_matrix(np.zeros([n_i, n_rn]))
    mat_net = lil_matrix(np.zeros([n_rn, n_rn]))
    mat_out = lil_matrix(np.zeros([n_rn, n_o]))

    i = 0
    for (_n, n_) in l_edges:
        if 'input' in _n:
            if weights == 'random':
                v = np.random.randint(1, 100)
            else:
                v = weights[i]

            mat_in[int(_n.split('_')[1]), int(n_.split('_')[1])] = v

        elif 'network' in _n:
            if 'network' in n_:
                if weights == 'random':
                    v = np.random.randint(1, 100)
                else:
                    v = weights[i]

                mat_net[int(_n.split('_')[1]), int(n_.split('_')[1])] = v

            elif 'output' in n_:
                if weights == 'random':
                    v = np.random.randint(1, 100)
                else:
                    v = weights[i]

                mat_out[int(_n.split('_')[1]), int(n_.split('_')[1])] = v

        i += 1

    return mat_in, mat_net, mat_out







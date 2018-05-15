# Global imports
import numpy as np
import unittest
from scipy.sparse import lil_matrix
from deyep.utils.names import KVName

# Local import
from deyep.core.constructors.constructors import Constructor, set_nodes, set_frequencies

__maintainer__ = 'Pierre Gouedard'


class TestConstructor(unittest.TestCase):
    def setUp(self):

        # Create a simple deep network (2 input nodes, 3 network nodes,, 2 output nodes)
        n_i, n_rn, n_o, self.capacity = 2, 5, 3, 2
        l_edges = [('input_0', 'network_0'), ('network_0', 'network_1'), ('network_0', 'network_2'),
                   ('network_1', 'output_0'), ('network_2', 'output_1')] +  \
                  [('input_0', 'network_3'), ('network_3', 'network_2')] + \
                  [('input_1', 'network_4'), ('network_4', 'output_2')] + \
                  [('input_1', 'network_2')]

        # Get matrices from list of edges and build network
        self.mat_in, self.mat_net, self.mat_out = get_mat_from_path(l_edges, n_i, n_rn, n_o)

    def test_deepnet_building(self):
        """
        Test basic frequency management
        python -m unittest tests.core.constructors.TestConstructor.test_deepnet_building

        """

        # Build dictionary of nodes
        d_inputs, d_net_ = set_nodes(self.mat_in, 'input')
        d_outputs, d_net_ = set_nodes(self.mat_out, 'output', d_net_nodes=d_net_)
        d_networks, _ = set_nodes(self.mat_net, 'network', d_net_nodes=d_net_)

        self.assertEqual(len(d_inputs), self.mat_in.shape[0])
        self.assertEqual(len(d_networks), self.mat_net.shape[0])
        self.assertEqual(len(d_outputs), self.mat_out.shape[-1])

        # Distribute frequencies
        d_inputs, available_freqs = set_frequencies(d_inputs, {0}, 1)
        d_networks, available_freqs = set_frequencies(d_networks, available_freqs, self.capacity, l_=d_inputs.values())

        # Assert that no frequency are conflicting
        import IPython
        IPython.embed()


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


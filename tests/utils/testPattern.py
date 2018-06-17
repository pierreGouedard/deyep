

# Global imports
from scipy.sparse import csc_matrix
import numpy as np

# local import
import settings
from deyep.core import nodes
from deyep.core.tools.frequencies import FrequencyStack


class TestPattern(object):

    def __init__(self, name, input_nodes, network_nodes, output_nodes):

        self.name = name
        self.input_nodes = input_nodes
        self.network_nodes = network_nodes
        self.output_nodes = output_nodes

    def build_graph_pattern(self):
        raise NotImplementedError

    def build_deterministic_io(self):
        raise NotImplementedError

    def build_random_io(self):
        raise NotImplementedError

    def generate_io_sequence(self, length):
        raise NotImplementedError


class TreePatter(TestPattern):

    def __init__(self, d, k):

        self.d = d
        self.k = k

        # Build list of nodess
        input_nodes = ['input_{}'.format(i) for i in range(self.get_size(d, k))]
        network_nodes = ['network_{}'.format(i) for i in range(self.get_size(d, k))]
        output_nodes = ['output_{}'.format(i) for i in range(self.get_size(d, k))]

        TestPattern.__init__(self, 'tree', input_nodes, network_nodes, output_nodes)




    @staticmethod
    def get_size(d, k):
        return sum([pow(k, i + 1) for i in range(d)])

    def build_graph_pattern(self):

        # Spread network nodes through depth
        d_nodes ={d + 2: range(pow(self.k, d + 2) - 2, pow(self.k, d + 3) - 2) for d in range(self.d - 1)}

        # Build input edges
        l_edges = []
        for i in range(self.k):
            l_edges += [(self.input_nodes[0], self.network_nodes[i])]

        for k, v in sorted(d_nodes.items(), key=lambda (k_, v_): k)[:-1]:
            for n, i in enumerate(sorted(v)):
                for j in range(self.k):
                    n_child = sorted(d_nodes[k + 1])[j + (n * self.k)]
                    l_edges += [(self.network_nodes[i], self.network_nodes[n_child])]

        # Build output edges
        l_edges += zip(self.network_nodes, self.output_nodes)

        return l_edges


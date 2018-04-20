# Global imports
import copy

# local import
import settings
from deyep.core import nodes
from deyep.core.tools.frequencies import FrequencyStack

class Constructors(object):

    def __init__(self, project, seed, feature_size, edge_density, w0):

        # settings
        self.project = project
        self.seed = seed

        # Params
        self.feature_size = feature_size
        self.edge_density = edge_density
        self.w0 = w0

        # Nodes
        self.input_nodes = []
        self.output_nodes = []
        self.network_nodes = []

    def from_weighted_direct_matrices(self, mat_net, mat_in, mat_out, capacity):

        # Get size of the deep network
        n_i, n_rn, n_o = mat_in.shape[0], mat_net.shape[0], mat_out.shape()[1]

        input_nodes, output_nodes, network_nodes = [], [], []
        xcoord, y_coord = mat_in.nonzero()
        for i in set(xcoord):
            node = nodes.InputNode('i{}'.format(i), None, None, FrequencyStack((n_i + n_rn) * capacity, i, capacity), [])
            for j in y_coord[xcoord == i]:
                node.children += [('n{}'.format(j), x)]

            input_nodes += [copy.deepcopy(node)]

    def build_input_nodes(self):
        raise NotImplementedError

    def build_output_nodes(self):
        raise NotImplementedError

    def build_network_nodes(self):
        raise NotImplemented

    def save_network(self, url):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError
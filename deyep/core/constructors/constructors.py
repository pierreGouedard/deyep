# Global imports
import copy

# local import
import settings
from deyep.core import nodes
from deyep.core.tools.frequencies import FrequencyStack


class Constructor(object):

    def __init__(self, project, seed, feature_size, edge_density, w0,
                 input_nodes=None, output_nodes=None, network_nodes=None):

        # settings
        self.project = project
        self.seed = seed

        # Params
        self.feature_size = feature_size
        self.edge_density = edge_density
        self.w0 = w0

        # Nodes
        self.input_nodes = [] if input_nodes is None else input_nodes
        self.output_nodes = [] if output_nodes is None else output_nodes
        self.network_nodes = [] if network_nodes is None else network_nodes

        # wait a minute
        self.deep_network = None

    @staticmethod
    def from_weighted_direct_matrices(mat_net, mat_in, mat_out, capacity, project=None, seed=None, feature_size=None,
                                      edge_density=None, w0=None):

        # Get size of the deep network
        n_i, n_rn, n_o = mat_in.shape[0], mat_net.shape[0], mat_out.shape[1]
        input_nodes, d_net_nodes = set_nodes(mat_in, 'input', n_i + n_rn, capacity, n_i)
        output_nodes, d_net_nodes = set_nodes(mat_out, 'output', n_i + n_rn, capacity, n_i, d_net_nodes=d_net_nodes)
        network_nodes, _ = set_nodes(mat_net, 'network', n_i + n_rn, capacity, n_i, d_net_nodes=d_net_nodes)

        return Constructor(project, seed, feature_size, edge_density, w0, input_nodes=input_nodes,
                           output_nodes=output_nodes, network_nodes=network_nodes)

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


def set_nodes(mat, key, N, c, offset, d_net_nodes={}):
    l_nodes = []
    d_net_nodes = d_net_nodes

    if key == 'input':
        x_coord, y_coord = mat.nonzero()
        for i in set(x_coord):
            node = nodes.InputNode(i, 'input', FrequencyStack(N * c, i, c), [])
            for j in y_coord[x_coord == i]:
                node.children += [('network_{}'.format(j), mat[i, j])]

                if j not in d_net_nodes.keys():
                    d_net_nodes[j] = nodes.NetworkNode(j, 'network', FrequencyStack(N * c, offset + j, c), [])

            l_nodes += [copy.deepcopy(node)]

    elif key == 'output':
        x_coord, y_coord = mat.nonzero()
        for j in set(y_coord):
            node = nodes.OutputNode(j, 'output', [])
            for i in x_coord[y_coord == j]:
                node.parents += [('network_{}'.format(i), mat[i, j])]

                if i not in d_net_nodes.keys():
                    d_net_nodes[i] = nodes.NetworkNode(i, 'network', FrequencyStack(N * c, offset + i, c), [])

            l_nodes += [copy.deepcopy(node)]

    elif key == 'network':
        x_coord, y_coord = mat.nonzero()
        for i in set(x_coord):

            if i in d_net_nodes.keys():
                d_net_nodes.pop(i)

            node = nodes.NetworkNode(i, 'network', FrequencyStack(N * c, offset + i, c), [])
            for j in y_coord[x_coord == i]:
                node.children += [('network_{}'.format(j), mat[i, j])]

            l_nodes += [copy.deepcopy(node)]

        l_nodes += d_net_nodes.values()

    else:
        raise ValueError('Choose key between \'input\', \'output\' or \'network\'')

    return l_nodes, d_net_nodes


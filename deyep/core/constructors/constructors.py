# Global imports
from scipy.sparse import csc_matrix
import numpy as np

# local import
import settings
from deyep.core import nodes
from deyep.core.tools.frequencies import FrequencyStack


class Constructor(object):

    def __init__(self, project, w0, edge_density=None, feature_size=None, seed=None, input_nodes=None, output_nodes=None,
                 network_nodes=None, deep_network=None):

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

        # Get wieght directed matrix graph
        if deep_network is not None:
            self.deep_network = deep_network
        else:
            self.deep_network = self.get_deep_network_from_nodes()

    @property
    def D(self):
        return csc_matrix((np.ones(len(self.Dw.data)), self.Dw.indices, self.Dw.indptr), shape=self.Dw.shape)

    @property
    def Dw(self):
        return self.deep_network['Dw']

    @property
    def O(self):
        return csc_matrix((np.ones(len(self.Ow.data)), self.Ow.indices, self.Ow.indptr), shape=self.Ow.shape)

    @property
    def Ow(self):
        return self.deep_network['Ow']

    @property
    def I(self):
        return csc_matrix((np.ones(len(self.Iw.data)), self.Iw.indices, self.Iw.indptr), shape=self.Iw.shape)

    @property
    def Iw(self):
        return self.deep_network['Iw']

    @property
    def Cm(self):
        return self.deep_network['Cm']

    @staticmethod
    def from_weighted_direct_matrices(mat_net, mat_in, mat_out, capacity, project=None, seed=None, edge_density=None,
                                      w0=None):

        # Get dict of nodes
        d_inputs, d_net_ = set_nodes(mat_in, 'input')
        d_outputs, d_net_ = set_nodes(mat_out, 'output', d_net_nodes=d_net_)
        d_networks, _ = set_nodes(mat_net, 'network', d_net_nodes=d_net_)

        # distribute frequency among input nodes,
        d_inputs, set_freqs = set_frequencies(d_inputs, {0}, 1)

        # distribute frequencies among network nodes
        d_networks, set_freqs = set_frequencies(d_networks, set_freqs, capacity, l_=d_inputs.values())

        # Finally, build nodes
        l_inputs = [nodes.InputNode(k_, 'input', FrequencyStack(len(set_freqs) * 2, v_['freqs']), v_['children'])
                    for k_, v_ in sorted(d_inputs.items(), key=lambda (k, v): k)]
        l_networks = [nodes.NetworkNode(k_, 'network', FrequencyStack(len(set_freqs) * 2, v_['freqs']), v_['children'])
                      for k_, v_ in sorted(d_networks.items(), key=lambda (k, v): k)]
        l_outputs = [nodes.OutputNode(k_, 'output', v_['parents'])
                     for k_, v_ in sorted(d_outputs.items(), key=lambda (k, v): k)]

        return Constructor(project, w0, input_nodes=l_inputs, output_nodes=l_outputs, network_nodes=l_networks,
                           deep_network={'Iw': mat_in, 'Dw': mat_net, 'Ow': mat_out, 'Cm': mat_out})

    def get_deep_network_from_nodes(self):
        raise NotImplementedError

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


def set_nodes(mat, key, d_net_nodes={}):
    d_nodes = {}
    d_net_nodes = d_net_nodes

    if key == 'input':
        x_coord, y_coord = mat.nonzero()
        for i in set(x_coord):
            d_nodes[i] = {'children': [], 'freqs':[]}
            for j in y_coord[x_coord == i]:
                d_nodes[i]['children'] += [('network_{}'.format(j), mat[i, j])]

                if j not in d_net_nodes.keys():
                    d_net_nodes[j] = {'children': [], 'freqs':[]}

    elif key == 'output':
        x_coord, y_coord = mat.nonzero()
        for j in set(y_coord):
            d_nodes[j] = {'parents': [], 'freqs':[]}
            for i in x_coord[y_coord == j]:
                d_nodes[j]['parents'] += [('network_{}'.format(i), mat[i, j])]

                if i not in d_net_nodes.keys():
                    d_net_nodes[i] = {'children': [], 'freqs':[]}

    elif key == 'network':
        x_coord, y_coord = mat.nonzero()
        for i in set(x_coord):

            if i in d_net_nodes.keys():
                d_net_nodes.pop(i)
            d_nodes[i] = {'children': [], 'freqs':[]}
            for j in y_coord[x_coord == i]:
                d_nodes[i]['children'] += [('network_{}'.format(j), mat[i, j])]

        d_nodes.update(d_net_nodes)

    else:
        raise ValueError('Choose key between \'input\', \'output\' or \'network\'')

    return d_nodes.copy(), d_net_nodes


def set_frequencies(d_nodes, freqs, capacity, l_=None):

    for k, v in d_nodes.items():

        l_children = [t[0] for t in v['children']]

        if len(l_children) == 0:
            freqs_ = list(freqs)[:capacity]

        else:
            # Look for sibling
            for child in l_children:
                _freqs = filter(lambda (k_, v_): child in [t[0] for t in v_['children']] and k != k_, d_nodes.items())
                _freqs = sum(map(lambda (k_, v_): v_['freqs'], _freqs), [])

                if l_ is not None:
                    _freqs += sum(map(lambda v_: v_['freqs'], filter(lambda v_: child in [t[0] for t in v_['children']],
                                                                     l_)), [])

                freqs_ = freqs.difference(set(_freqs))

        if len(freqs_) >= capacity:
            d_nodes[k]['freqs'] = list(freqs_)[:capacity]

        else:
            # Compute new frequencies
            new_freqs = range(max(freqs) + 1, max(freqs) + capacity + 1 - len(freqs_))

            # Update available freq
            freqs = freqs.union(set(new_freqs))

            d_nodes[k]['freqs'] = list(set(new_freqs).union(set(freqs_)))

    return d_nodes, freqs

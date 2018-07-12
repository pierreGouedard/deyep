# Global imports
from scipy.sparse import csc_matrix
import numpy as np

# local import
from deyep.core.builder import comon


class DeepNetwork(object):

    def __init__(self, w0, l0, tau, input_nodes, output_nodes, network_nodes, graph, n_freq, seed=None):

        self.seed = seed
        self.w0 = w0
        self.l0 = l0
        self.tau = tau

        # Nodes
        self.n_freq = n_freq
        self.input_nodes = [] if input_nodes is None else input_nodes
        self.output_nodes = [] if output_nodes is None else output_nodes
        self.network_nodes = [] if network_nodes is None else network_nodes

        # Get matrices involved in graph
        self.graph = graph

    @property
    def D(self):
        return csc_matrix((np.ones(len((self.Dw > 0).data)), (self.Dw > 0).indices, (self.Dw > 0).indptr),
                          shape=self.Dw.shape)

    @property
    def Dw(self):
        return self.graph['Dw']

    @property
    def O(self):
        return csc_matrix((np.ones(len((self.Ow > 0).data)), (self.Ow > 0).indices, (self.Ow > 0).indptr),
                          shape=self.Ow.shape)

    @property
    def Ow(self):
        return self.graph['Ow']

    @property
    def I(self):
        return csc_matrix((np.ones(len((self.Iw > 0).data)), (self.Iw > 0).indices, (self.Iw > 0).indptr),
                          shape=self.Iw.shape)

    @property
    def Iw(self):
        return self.graph['Iw']

    @property
    def Cm(self):
        return self.graph['Cm']

    @staticmethod
    def from_matrices(sax_net, sax_in, sax_out, capacity, seed=None, w0=5, l0=10, tau=5):
        l_inputs, l_outputs, l_networks, n_freq = comon.nodes_from_mat(sax_net, sax_in, sax_out, capacity, l0=l0)
        d_graph = {'Iw': sax_in, 'Dw': sax_net, 'Ow': sax_out, 'Cm': sax_out}

        return DeepNetwork(w0, l0, tau, input_nodes=l_inputs, output_nodes=l_outputs, network_nodes=l_networks,
                           n_freq=n_freq, graph=d_graph, seed=seed)

    @staticmethod
    def from_nodes(w0, input_nodes, output_nodes, network_nodes, seed=None):
        raise NotImplementedError

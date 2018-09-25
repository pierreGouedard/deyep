# Global imports
import random
import string
import pickle

# local import
from deyep.core.builder import comon
import settings
from deyep.utils.driver.driver import FileDriver
from deyep.core.tools.basis.canonical import CanonicalBasis
from deyep.core.tools.basis.fourrier import FourrierBasis
driver = FileDriver('network_file_driver', '')


class DeepNetwork(object):

    def __init__(self, project, w0, l0, tau, input_nodes, output_nodes, network_nodes, graph, n_freq, network_id=None):

        if network_id is None:
            network_id = random.choice(string.ascii_letters)

        self.pth_network = driver.join(settings.deyep_imputer_path.format(project), '{}.pckl'.format(network_id))
        self.network_id = network_id

        # Paramter
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
        return self.Dw > 0

    @property
    def Dw(self):
        return self.graph['Dw']

    @property
    def O(self):
        return self.Ow > 0

    @property
    def Ow(self):
        return self.graph['Ow']

    @property
    def I(self):
        return self.Iw > 0

    @property
    def Iw(self):
        return self.graph['Iw']

    @property
    def Cm(self):
        return self.graph['Cm']

    @staticmethod
    def from_matrices(project, sax_net, sax_in, sax_out, capacity, basis, network_id=None, w0=5, l0=10, tau=5):
        l_inputs, l_outputs, l_networks, n_freq = comon.nodes_from_mat(sax_net, sax_in, sax_out, capacity, basis, l0=l0)
        d_graph = {'Iw': sax_in, 'Dw': sax_net, 'Ow': sax_out, 'Cm': sax_out}

        return DeepNetwork(project, w0, l0, tau, input_nodes=l_inputs, output_nodes=l_outputs, network_nodes=l_networks,
                           n_freq=n_freq, graph=d_graph, network_id=network_id)

    @staticmethod
    def from_nodes(w0, input_nodes, output_nodes, network_nodes, seed=None):
        raise NotImplementedError

    @staticmethod
    def from_dict(project, d_network, network_id=None, basis='canonical'):
        d_basis = {'canonical': CanonicalBasis, 'fourrier': FourrierBasis}
        dn = DeepNetwork(
            project, d_network['w0'], d_network['l0'], d_network['tau'], graph=d_network['graph'],
            n_freq=d_network['n_freq'], input_nodes=[n.from_dict(d_basis[basis]) for n in d_network['input_nodes']],
            output_nodes=[n.from_dict() for n in d_network['output_nodes']],
            network_nodes=[n.from_dict(d_basis[basis]) for n in d_network['network_nodes']],
            network_id=network_id
        )

        return dn

    def to_dict(self):
        d_network = {'graph': self.graph, 'w0': self.w0, 'l0': self.l0, 'tau': self.tau, 'n_freq': self.n_freq,
                     'input_nodes': [n.to_dict() for n in self.input_nodes],
                     'network_nodes': [n.to_dict() for n in self.network_nodes],
                     'output_nodes': [n.to_dict() for n in self.output_nodes]}
        return d_network

    def save(self):
        d_network = self.to_dict()

        with open(self.pth_network, 'wb') as handle:
            pickle.dump(d_network, handle)

    @staticmethod
    def load(project, network_id, basis='canonical'):
        pth = driver.join(settings.deyep_network_path.format(project), '{}.pckl'.format(network_id))

        with open(pth, 'rb') as handle:
            d_network = pickle.load(handle)

        return DeepNetwork.from_dict(project, d_network, network_id=network_id, basis=basis)

# Global imports
import pickle
import random
import string

import numpy as np

import settings
from deyep.core.builder import comon
from deyep.core.datastructures.nodes import InputNode as ni, OutputNode as no, NetworkNode as nd
from deyep.core.tools.basis.comon import Basis
from deyep.utils.driver.driver import FileDriver

driver = FileDriver('network_file_driver', '')


class DeepNetwork(object):

    def __init__(self, project, w0, l0, tau, input_nodes, output_nodes, network_nodes, graph, n_freq, capacity,
                 network_id=None, is_fitted=False):

        if network_id is None:
            network_id = ''.join([random.choice(string.ascii_letters) for _ in range(5)])

        self.project = project
        self.dir_network = settings.deyep_network_path.format(project)
        self.network_id = network_id
        self.is_dried = False
        self.is_fitted = is_fitted

        # Parameter
        self.w0 = w0
        self.l0 = l0
        self.tau = tau

        # Nodes
        self.n_freq = n_freq
        self.node_capacity = capacity
        self.input_nodes = [] if input_nodes is None else input_nodes
        self.output_nodes = [] if output_nodes is None else output_nodes
        self.network_nodes = [] if network_nodes is None else network_nodes

        # Get matrices involved in graph and add nodes stats
        self.graph = graph
        self.graph.update({'N_f': np.array([0] * len(network_nodes), dtype=int),
                           'N_p': np.array([0] * len(network_nodes)),
                           'N_r': np.array([0] * len(network_nodes))})

        self.d_metrics = None

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

    @property
    def IOw(self):
        return self.graph.get('IOw', None)

    @staticmethod
    def load(project, network_id):
        pth = driver.join(settings.deyep_network_path.format(project), '{}.pckl'.format(network_id))

        with open(pth, 'rb') as handle:
            d_network = pickle.load(handle)

        return DeepNetwork.from_dict(d_network, project, network_id=network_id)

    @staticmethod
    def from_matrices(project, sax_net, sax_in, sax_out, capacity, network_id=None, w0=5, l0=10, tau=5, Cm=None,
                      sax_io=None, levels=None):

        l_inputs, l_outputs, l_networks, n_freq = \
            comon.nodes_from_mat(sax_net, sax_in, sax_out, capacity, l0=l0, levels=levels)

        d_graph = {'Iw': sax_in, 'Dw': sax_net, 'Ow': sax_out, 'Cm': {None: sax_out}.get(Cm, Cm),
                   'IOw': sax_io}

        return DeepNetwork(project, w0, l0, tau, input_nodes=l_inputs, output_nodes=l_outputs, network_nodes=l_networks,
                           n_freq=n_freq, graph=d_graph, capacity=capacity, network_id=network_id)

    @staticmethod
    def from_dict(d_network, project, network_id=None):
        dn = DeepNetwork(
            project, d_network['w0'], d_network['l0'], d_network['tau'], graph=d_network['graph'],
            n_freq=d_network['n_freq'], capacity=d_network['node_capacity'],
            input_nodes=[ni.from_dict(d_n, Basis) for d_n in d_network['input_nodes']],
            output_nodes=[no.from_dict(d_n) for d_n in d_network['output_nodes']],
            network_nodes=[nd.from_dict(d_n, Basis) for d_n in d_network['network_nodes']],
            network_id=network_id, is_fitted=d_network['is_fitted']
        )

        return dn

    def copy(self):
        dn_ = DeepNetwork.from_matrices(
            self.project, self.Dw.multiply(self.D), self.Iw.multiply(self.I), self.Ow.multiply(self.O),
            self.node_capacity, network_id='copy_{}'.format(self.network_id), w0=self.w0, l0=self.l0, tau=self.tau,
            Cm=self.Cm, sax_io=self.IOw, levels=[n.d_levels for n in self.network_nodes],
        )
        return dn_

    def save(self):
        if not driver.exists(self.dir_network):
            driver.makedirs(self.dir_network)

        d_network = self.to_dict()

        with open(driver.join(self.dir_network, '{}.pckl'.format(self.network_id)), 'wb') as handle:
            pickle.dump(d_network, handle)

    def to_dict(self):
        d_network = {'node_capacity': self.node_capacity,
                     'is_fitted': self.is_fitted,
                     'is_dried': False,
                     'graph': self.graph, 'w0': self.w0, 'l0': self.l0, 'tau': self.tau, 'n_freq': self.n_freq,
                     'input_nodes': [n.to_dict() for n in self.input_nodes],
                     'network_nodes': [n.to_dict() for n in self.network_nodes],
                     'output_nodes': [n.to_dict() for n in self.output_nodes]}
        return d_network

    def delete(self):
        if driver.exists(driver.join(self.dir_network, '{}.pckl'.format(self.network_id))):
            driver.remove(driver.join(self.dir_network, '{}.pckl'.format(self.network_id)))

    def set_metrics(self, depth, ax_got, ax_out):
        d_metrics = {}

        # Compute precision
        d_metrics['P'] = self.compute_precision(ax_got, ax_out)

        # Compute recall
        d_metrics['R'] = self.compute_recall(ax_got, ax_out)

        # Compute efficiency
        d_metrics['E'] = self.compute_efficiency(depth)

        self.d_metrics = d_metrics

    def compute_precision(self, ax_got, ax_out):
        return float((ax_got & ax_out).sum()) / (ax_out.sum() + 1e-6)

    def compute_recall(self, ax_got, ax_out):
        return float((ax_got & ax_out).sum()) / (ax_got.sum() + 1e-6)

    def compute_efficiency(self, depth):
        return float(len(self.network_nodes)) / (len(self.input_nodes) * depth)

    @staticmethod
    def reduce_network(dn, ax_active_nodes):

        # Now clean matrices  of the deep network
        sax_I = dn.Iw.multiply(dn.I)[:, ax_active_nodes]
        sax_D = dn.Dw.multiply(dn.D)[:, ax_active_nodes][ax_active_nodes, :]
        sax_O = dn.Ow.multiply(dn.O)[ax_active_nodes, :]
        sax_Cm = dn.Cm[ax_active_nodes, :]
        levels = [n.d_levels for i, n in enumerate(dn.network_nodes) if ax_active_nodes[i]]

        # Build new deep network
        deep_network = DeepNetwork.from_matrices(dn.project, sax_D, sax_I, sax_O, dn.node_capacity,
                                                 network_id=dn.network_id, w0=dn.w0, l0=dn.l0, tau=dn.tau, Cm=sax_Cm,
                                                 levels=levels)

        return deep_network



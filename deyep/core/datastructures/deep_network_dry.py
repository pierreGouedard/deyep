# Global imports
import pickle
import random
import string

# Local import
import settings
from deyep.core.builder import comon
from deyep.core.datastructures.nodes_dry import InputNodeDry as ni, OutputNodeDry as no, NetworkNodeDry as nd
from deyep.utils.driver.driver import FileDriver

driver = FileDriver('network_file_driver', '')


class DeepNetworkDry(object):

    def __init__(self, project, input_nodes, output_nodes, network_nodes, graph, network_id=None):

        if network_id is None:
            network_id = 'dry_{}'.join([random.choice(string.ascii_letters) for _ in range(7)])

        self.project = project
        self.dir_network = settings.deyep_network_path.format(project)
        self.network_id = network_id
        self.is_dried = True

        # Nodes
        self.input_nodes = [] if input_nodes is None else input_nodes
        self.output_nodes = [] if output_nodes is None else output_nodes
        self.network_nodes = [] if network_nodes is None else network_nodes

        # Get matrices involved in graph and add nodes stats
        self.graph, self.d_metrics = graph, None

    @property
    def D(self):
        return self.graph['D']

    @property
    def O(self):
        return self.graph['O']

    @property
    def I(self):
        return self.graph['I']

    def copy(self):
        dn_ = DeepNetworkDry.from_matrices(
            self.project, self.D, self.I, self.O, network_id='copy_{}'.format(self.network_id),
            levels=[n.d_levels for n in self.network_nodes]
        )
        return dn_

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
        return float((ax_got & ax_out).sum()) / ax_out.sum()

    def compute_recall(self, ax_got, ax_out):
        return float((ax_got & ax_out).sum()) / ax_got.sum()

    def compute_efficiency(self, depth):
        return float(len(self.network_nodes)) / (len(self.input_nodes) * depth)

    @staticmethod
    def from_matrices(project, sax_net, sax_in, sax_out, network_id=None, levels=None):
        l_inputs, l_outputs, l_networks = comon.nodes_from_mat_dry(sax_net, sax_in, sax_out, levels=levels)
        d_graph = {'I': sax_in, 'D': sax_net, 'O': sax_out}

        return DeepNetworkDry(project, input_nodes=l_inputs, output_nodes=l_outputs,
                              network_nodes=l_networks, graph=d_graph, network_id=network_id)

    @staticmethod
    def reduce_network(dn, ax_active_nodes):

        # Now clean matrices  of the deep network
        sax_I = dn.I[:, ax_active_nodes]
        sax_D = dn.D[:, ax_active_nodes][ax_active_nodes, :]
        sax_O = dn.O[ax_active_nodes, :]
        levels = [n.d_levels for i, n in enumerate(dn.network_nodes) if ax_active_nodes[i]]

        # Build new deep network
        dnd = DeepNetworkDry.from_matrices(dn.project, sax_D, sax_I, sax_O, network_id=dn.network_id, levels=levels)

        return dnd

    @staticmethod
    def from_deep_network(dn):

        if dn.is_dried:
            return dn
        else:
            levels = [{k: v > dn.tau for k, v in n.d_levels.items()} for n in dn.network_nodes]

        l_inputs, l_outputs, l_networks = comon.nodes_from_mat_dry(dn.D, dn.I, dn.O, levels=levels)
        d_graph = {'I': dn.I, 'D': dn.D, 'O': dn.O}

        return DeepNetworkDry(dn.project, input_nodes=l_inputs, output_nodes=l_outputs, network_nodes=l_networks,
                              graph=d_graph, network_id='dry_{}'.format(dn.network_id))

    @staticmethod
    def from_dict(d_network, project, network_id=None):
        dn = DeepNetworkDry(
            project, graph=d_network['graph'],
            input_nodes=[ni.from_dict(d_n) for d_n in d_network['input_nodes']],
            output_nodes=[no.from_dict(d_n) for d_n in d_network['output_nodes']],
            network_nodes=[nd.from_dict(d_n) for d_n in d_network['network_nodes']],
            network_id=network_id
        )

        return dn

    def to_dict(self):
        d_network = {'graph': self.graph, 'is_dried': True,
                     'input_nodes': [n.to_dict() for n in self.input_nodes],
                     'network_nodes': [n.to_dict() for n in self.network_nodes],
                     'output_nodes': [n.to_dict() for n in self.output_nodes]}
        return d_network

    def save(self):
        if not driver.exists(self.dir_network):
            driver.makedirs(self.dir_network)

        d_network = self.to_dict()

        with open(driver.join(self.dir_network, '{}.pckl'.format(self.network_id)), 'wb') as handle:
            pickle.dump(d_network, handle)

    def delete(self):
        if driver.exists(driver.join(self.dir_network, '{}.pckl'.format(self.network_id))):
            driver.remove(driver.join(self.dir_network, '{}.pckl'.format(self.network_id)))

    @staticmethod
    def load(project, network_id):
        pth = driver.join(settings.deyep_network_path.format(project), '{}.pckl'.format(network_id))

        with open(pth, 'rb') as handle:
            d_network = pickle.load(handle)

        return DeepNetworkDry.from_dict(d_network, project, network_id=network_id)

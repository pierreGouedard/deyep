# Global imports
import pickle
import random
import string

# Local import
import settings
from deyep.core.firing_graph import utils
from deyep.core.firing_graph.vertex import InputVertex as iv, OutputVertex as ov, CoreVertex as cv
from deyep.core.tools.basis.comon import Basis
from deyep.utils.driver.driver import FileDriver

driver = FileDriver('graph_file_driver', '')


class FiringGraph(object):
    """
    This class implement the main data structure used for fitting data. It is composed of different class of vertices
    and store weighted connexion in the form of scipy.sparse matrices.

    """
    def __init__(self, project, w0, ax_levels, input_vertices, output_vertices, core_vertices, graph, n_freq, capacity,
                 graph_id=None, is_drained=False):

        if graph_id is None:
            graph_id = ''.join([random.choice(string.ascii_letters) for _ in range(5)])

        self.project = project
        self.dir_graph = settings.deyep_graph_path.format(project)
        self.graph_id = graph_id
        self.is_dried = False
        self.is_drained = is_drained

        # Parameter
        self.w0 = w0
        self.levels = ax_levels

        # Nodes
        self.n_freq = n_freq
        self.vertex_capacity = capacity
        self.input_vertices = [] if input_vertices is None else input_vertices
        self.output_vertices = [] if output_vertices is None else output_vertices
        self.core_vertices = [] if core_vertices is None else core_vertices

        # Get matrices involved in graph and add nodes stats
        self.graph = graph
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

    @staticmethod
    def load(project, graph_id):
        pth = driver.join(settings.deyep_graph_path.format(project), '{}.pckl'.format(graph_id))

        with open(pth, 'rb') as handle:
            d_graph = pickle.load(handle)

        return FiringGraph.from_dict(d_graph, project, graph_id=graph_id)

    @staticmethod
    def from_matrices(project, sax_D, sax_I, sax_O, capacity, w0, ax_levels, graph_id=None):

        l_inputs, l_outputs, l_cores, n_freq = utils.vertices_from_mat(
            sax_D, sax_I, sax_O, capacity, ax_levels
        )

        d_graph = {'Iw': sax_I, 'Dw': sax_D, 'Ow': sax_O}

        return FiringGraph(
            project, w0, ax_levels, input_vertices=l_inputs, output_vertices=l_outputs, core_vertices=l_cores,
            n_freq=n_freq, graph=d_graph, capacity=capacity, graph_id=graph_id
        )

    @staticmethod
    def from_dict(d_graph, project, graph_id=None):
        fg = FiringGraph(
            project, d_graph['w0'], d_graph['levels'], graph=d_graph['graph'],
            n_freq=d_graph['n_freq'], capacity=d_graph['cv_capacity'],
            input_vertices=[iv.from_dict(d_n, Basis) for d_n in d_graph['input_vertices']],
            output_vertices=[ov.from_dict(d_n) for d_n in d_graph['output_vertices']],
            core_vertices=[cv.from_dict(d_n, Basis) for d_n in d_graph['core_vertices']],
            graph_id=graph_id, is_drained=d_graph['is_drained']
        )

        return fg

    def copy(self):
        fg_ = FiringGraph.from_matrices(
            self.project, self.Dw.multiply(self.D), self.Iw.multiply(self.I), self.Ow.multiply(self.O),
            self.vertex_capacity, graph_id=self.graph_id, w0=self.w0, ax_levels=self.levels,
        )
        return fg_

    def save(self):
        if not driver.exists(self.dir_graph):
            driver.makedirs(self.dir_graph)

        d_network = self.to_dict()

        with open(driver.join(self.dir_graph, '{}.pckl'.format(self.graph_id)), 'wb') as handle:
            pickle.dump(d_network, handle)

    def to_dict(self):
        d_network = {'vertex_capacity': self.vertex_capacity,
                     'is_drained': self.is_drained,
                     'is_dried': False,
                     'graph': self.graph, 'w0': self.w0, 'levels': self.levels, 'n_freq': self.n_freq,
                     'input_nodes': [n.to_dict() for n in self.input_vertices],
                     'network_nodes': [n.to_dict() for n in self.core_vertices],
                     'output_nodes': [n.to_dict() for n in self.output_vertices]}
        return d_network

    def delete(self):
        if driver.exists(driver.join(self.dir_graph, '{}.pckl'.format(self.graph_id))):
            driver.remove(driver.join(self.dir_graph, '{}.pckl'.format(self.graph_id)))

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
        return float(len(self.core_vertices)) / (len(self.input_vertices) * depth)

    @staticmethod
    def reduce_network(fg, ax_active_nodes):

        # Now clean matrices  of the deep network
        sax_I = fg.Iw.multiply(fg.I)[:, ax_active_nodes]
        sax_D = fg.Dw.multiply(fg.D)[:, ax_active_nodes][ax_active_nodes, :]
        sax_O = fg.Ow.multiply(fg.O)[ax_active_nodes, :]

        firing_graph = FiringGraph.from_matrices(
            fg.project, sax_D, sax_I, sax_O, fg.vertex_capacity, graph_id=fg.graph_id, w0=fg.w0, ax_levels=fg.ax_levels
        )

        return firing_graph



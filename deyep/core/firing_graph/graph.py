# Global imports
import pickle
import random
import string
from scipy.sparse import diags, lil_matrix

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
    def __init__(self, project, t, ax_levels, input_vertices, output_vertices, core_vertices, graph, mask_drain, n_freq,
                 graph_id=None, is_drained=False):

        if graph_id is None:
            graph_id = ''.join([random.choice(string.ascii_letters) for _ in range(5)])

        self.project = project
        self.dir_graph = settings.deyep_graph_path.format(project)
        self.graph_id = graph_id
        self.is_dried = False
        self.is_drained = is_drained

        # Parameter
        self.t = t
        self.levels = ax_levels

        # Nodes
        self.n_freq = n_freq
        self.input_vertices = [] if input_vertices is None else input_vertices
        self.output_vertices = [] if output_vertices is None else output_vertices
        self.core_vertices = [] if core_vertices is None else core_vertices

        # Store matrices
        self.graph = graph
        self.mask_vertices = mask_drain
        self.mask_mat = None
        self.update_mask_mat()

        self.backward_firing = {
            'ip': lil_matrix(graph['Iw'].shape), 'in': lil_matrix(graph['Iw'].shape),
            'cp': lil_matrix(graph['Dw'].shape), 'cn': lil_matrix(graph['Dw'].shape),
            'op': lil_matrix(graph['Ow'].shape), 'on': lil_matrix(graph['Ow'].shape),
        }

        self.forward_firing = {
            'i': lil_matrix((1, graph['Iw'].shape[0])), 'c': lil_matrix((1, graph['Dw'].shape[0])),
            'o': lil_matrix((1, graph['Ow'].shape[1]))
        }

    @property
    def D(self):
        return self.Dw > 0

    @property
    def Dw(self):
        return self.graph['Dw']

    @property
    def D_mask(self):
        return self.mask_mat['D']

    @property
    def O(self):
        return self.Ow > 0

    @property
    def Ow(self):
        return self.graph['Ow']

    @property
    def O_mask(self):
        return self.mask_mat['O']

    @property
    def I(self):
        return self.Iw > 0

    @property
    def Iw(self):
        return self.graph['Iw']

    @property
    def I_mask(self):
        return self.mask_mat['I']

    def update_policy(self, key):
        return self.mask_vertices.get(key, 'D').sum() > 0

    def update_mask_mat(self):
        self.mask_mat = {
            k: diags(ax_mask).dot(self.graph['{}w'.format(k)] > 0) for k, ax_mask in self.mask_vertices.items()
        }
        self.mask_mat.update({'O': diags(self.mask_vertices['D']).dot(self.graph['{}w'.format('O')] > 0)})

    def update_bakward_firing(self, sax_Iu, sax_Du, sax_Ou):
        # TODO update mask only if specified (costly op) and optimize operation
        if sax_Iu is not None:
            self.backward_firing['ip'] += (sax_Iu > 0)
            self.backward_firing['in'] += (sax_Iu < 0)
            self.mask_mat['I'] = \
                self.mask_mat['I'].multiply(self.backward_firing['ip'] + self.backward_firing['in'] < self.t)\
                    .multiply(self.I)

        if sax_Ou is not None and sax_Du is not None:
            self.backward_firing['cp'] += (sax_Du > 0)
            self.backward_firing['cn'] += (sax_Du < 0)
            self.mask_mat['D'] = \
                self.mask_mat['D'].multiply(self.backward_firing['cp'] + self.backward_firing['cn'] < self.t)\
                    .multiply(self.D)

            self.backward_firing['op'] += (sax_Ou > 0)
            self.backward_firing['on'] += (sax_Ou < 0)
            self.mask_mat['O'] = \
                self.mask_mat['O'].multiply(self.backward_firing['op'] + self.backward_firing['on'] < self.t)\
                    .multiply(self.O)

    def update_forward_firing(self, sax_i, sax_c, sax_o):
        print sax_i.toarray().sum(axis=0)[8]
        self.forward_firing['i'] += sax_i.sum(axis=0)
        self.forward_firing['c'] += sax_c.sum(axis=0)
        self.forward_firing['o'] += sax_o.sum(axis=0)

    @staticmethod
    def load(project, graph_id):
        pth = driver.join(settings.deyep_graph_path.format(project), '{}.pckl'.format(graph_id))

        with open(pth, 'rb') as handle:
            d_graph = pickle.load(handle)

        return FiringGraph.from_dict(d_graph, project, graph_id=graph_id)

    @staticmethod
    def from_matrices(project, sax_D, sax_I, sax_O, capacity, t, ax_levels, mask_drain, graph_id=None):

        l_inputs, l_outputs, l_cores, n_freq = utils.vertices_from_mat(
            sax_D, sax_I, sax_O, capacity, ax_levels
        )

        # Init dict of structure of firing graph
        d_graph = {'Iw': sax_I, 'Dw': sax_D, 'Ow': sax_O}

        return FiringGraph(
            project, t, ax_levels, input_vertices=l_inputs, output_vertices=l_outputs, core_vertices=l_cores,
            n_freq=n_freq, graph=d_graph, mask_drain=mask_drain, graph_id=graph_id
        )

    @staticmethod
    def from_dict(d_graph, project, graph_id=None):
        fg = FiringGraph(
            project, d_graph['N'], d_graph['levels'], graph=d_graph['graph'], mask_drain=d_graph['mask'],
            n_freq=d_graph['n_freq'], input_vertices=[iv.from_dict(d_n, Basis) for d_n in d_graph['input_vertices']],
            output_vertices=[ov.from_dict(d_n) for d_n in d_graph['output_vertices']],
            core_vertices=[cv.from_dict(d_n, Basis) for d_n in d_graph['core_vertices']],
            graph_id=graph_id, is_drained=d_graph['is_drained']
        )

        return fg

    def copy(self):

        capacity = self.core_vertices[0].basis.capacity

        fg_ = FiringGraph.from_matrices(
            self.project, self.Dw.multiply(self.D), self.Iw.multiply(self.I), self.Ow.multiply(self.O), capacity,
            graph_id=self.graph_id, t=self.t, ax_levels=self.levels, mask_drain=self.mask_vertices
        )
        return fg_

    def save(self):
        if not driver.exists(self.dir_graph):
            driver.makedirs(self.dir_graph)

        d_network = self.to_dict()

        with open(driver.join(self.dir_graph, '{}.pckl'.format(self.graph_id)), 'wb') as handle:
            pickle.dump(d_network, handle)

    def to_dict(self):
        d_network = {'is_drained': self.is_drained,
                     'is_dried': False,
                     'graph': self.graph, 't': self.t, 'levels': self.levels, 'mask': self.mask_vertices,
                     'n_freq': self.n_freq, 'input_nodes': [n.to_dict() for n in self.input_vertices],
                     'network_nodes': [n.to_dict() for n in self.core_vertices],
                     'output_nodes': [n.to_dict() for n in self.output_vertices]
                     }
        return d_network

    def delete(self):
        if driver.exists(driver.join(self.dir_graph, '{}.pckl'.format(self.graph_id))):
            driver.remove(driver.join(self.dir_graph, '{}.pckl'.format(self.graph_id)))

    @staticmethod
    def reduce_network(fg, ax_active_nodes):

        # Now clean matrices  of the deep network
        sax_I = fg.Iw.multiply(fg.I)[:, ax_active_nodes]
        sax_D = fg.Dw.multiply(fg.D)[:, ax_active_nodes][ax_active_nodes, :]
        sax_O = fg.Ow.multiply(fg.O)[ax_active_nodes, :]

        firing_graph = FiringGraph.from_matrices(
            fg.project, sax_D, sax_I, sax_O, fg.vertex_capacity, graph_id=fg.graph_id, t=fg.t,
            mask_drain=fg.mask_vertices, ax_levels=fg.ax_levels
        )

        return firing_graph



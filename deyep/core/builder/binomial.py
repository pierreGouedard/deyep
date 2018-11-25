# Global import
import numpy as np
from scipy.sparse import lil_matrix

# Local import
from deyep.core.builder.comon import mat_from_tuples
from deyep.core.datastructures.deep_network import DeepNetwork
from deyep.core.merger.comon import DeepNetMerger

class BinomialGraphBuilder(object):

    """
    Class that will build a graph based on a binomial method in theoretical report
    """
    def __init__(self, ni, nd, no, depth_max, p0, w0, l0, tau, capacity, basis='canonical', ax_p=None):

        # Core feature of graph
        self.ni, self.nd, self.nd_all, self.no, self.depth_max, self.p0, self.w0 = ni, nd, nd, no, depth_max, p0, w0

        # Core feature of nodes
        self.l0, self.tau, self.basis, self.capacity = l0, tau, basis, capacity

        # Init binomial construction
        self.l_edges = []
        self.gammas = [['input_{}'.format(i) for i in range(self.ni)]]
        self.delta = ['network_{}'.format(i) for i in range(self.nd)]

        # Build probas
        if ax_p is not None:
            self.ax_p = ax_p
        else:
            self.ax_p = BinomialGraphBuilder.init_depth_probablities(self.p0, self.depth_max)

    @staticmethod
    def init_depth_probablities(p0, depth_max):
        """

        :param p0:
        :param depth_max:
        :return:
        """
        ax_p = np.zeros([depth_max, depth_max])
        for i in range(depth_max):
            for j in range(i):
                ax_p[i, i - j] = p0 / (i * (i - j))

        return ax_p

    def build_tuples(self):
        """

        :return:
        """

        l_edges, gamma_, i, offset = [], set(), 1, 0
        while len(self.gammas[i - 1]) > 0 and self.depth_max > i:
            for j in range(i):
                l_edges, gamma_ = self.generate_random_link(i, j, gamma_, l_edges)

            self.gammas += [list(gamma_)]
            offset = max(map(lambda x: int(x.split('_')[-1]), list(gamma_))) + 1
            self.delta = ['network_{}'.format(k + offset) for k in range(self.nd)]
            gamma_ = set()
            i += 1

        # reindex network nodes
        self.reindex_nodes(l_edges)

    def generate_random_link(self, i, j, gamma_, l_edges):
        """

        :param i:
        :param j:
        :param gamma_:
        :param l_edges:
        :return:
        """
        p = self.ax_p[i, abs(i - j)]

        for nx in self.gammas[j]:
            for ny in self.delta:
                if np.random.choice([True, False], p=[p, 1. - p]):
                    l_edges += [(nx, ny)]
                    gamma_ = gamma_.union({ny})

        return l_edges, gamma_

    def reindex_nodes(self, l_edges):
        """

        :param l_edges:
        :return:
        """
        # Gather nodes
        l_nodes = sum(self.gammas[1:], [])

        # Build index
        d_index = dict([(k, 'network_{}'.format(i)) for i, k in enumerate(l_nodes)])

        # Transform edges and network nodes
        l_edges_ = [(nx, d_index[ny]) if 'input' in nx else (d_index[nx], d_index[ny]) for nx, ny in l_edges]

        gammas = [self.gammas[0]]
        for l_nodes in self.gammas[1:]:
            gammas += [[d_index[n] for n in l_nodes]]

        self.gammas = gammas
        self.l_edges = l_edges_
        self.nd_all = len(d_index.values())

    def build_matrices(self):
        """

        :return:
        """
        if len(self.l_edges) == 0:
            self.build_tuples()

        return mat_from_tuples(self.l_edges, self.ni, self.nd_all, self.no, weights=[self.w0] * len(self.l_edges))

    def build_network(self, project, network_id=None, delay=0):
        """

        :return:
        """

        if len(self.l_edges) == 0:
            self.build_tuples()

        (mat_in, mat_net, mat_out), sax_Cm = self.build_matrices(), None

        if delay > 0:
            sax_Cm = lil_matrix(mat_out.shape, dtype=bool)
            for l_nodes in self.gammas[1:delay + 1]:
                sax_Cm[[int(n.split('_')[1]) for n in l_nodes], :] = True

        return DeepNetwork.from_matrices(project, mat_net, mat_in, mat_out, self.capacity, self.basis, w0=self.w0,
                                         l0=self.l0, tau=self.tau, network_id=network_id, Cm=sax_Cm)

    def reset_structure(self):
        self.l_edges = []
        self.gammas = [['input_{}'.format(i) for i in range(self.ni)]]
        self.delta = ['network_{}'.format(i) for i in range(self.nd)]

    @staticmethod
    def layout(sax_I, sax_D, sax_O):

        # Get dimension of graph
        ni, nd, no = sax_I.shape[0], sax_D.shape[0], sax_O.shape[-1]
        d_nodes, _ = DeepNetMerger.classify_nodes_by_layer(sax_I, sax_D, range(sax_D.shape[0]))

        # Get Input position
        pos = {'inputs': {'pos': {i: (0, (nd - ni) / 2 + i) for i in range(ni)}, 'color': 'r'}}

        # Get network position
        pos['network'] = {'pos': {}, 'color': 'k'}
        for k, idx in d_nodes.items():
            pos['network']['pos'].update(
                {ni + i: (k + 1 + (np.random.randn() / 7), np.random.randint(0, nd)) for (i, _) in idx}
            )

        # Get Output position
        pos.update({'outputs': {'pos': {ni + nd + i: (max(d_nodes.keys()) + 2, (nd - no) / 2 + i) for i in range(no)},
                                'color': 'b'}})

        return pos

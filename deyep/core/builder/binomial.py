# Global import
import numpy as np
import networkx as nx

# Local import
from deyep.core.builder.comon import mat_from_tuples
from deyep.core.deep_network import DeepNetwork


class BinomialGraphBuilder(object):

    """
    Class that will build a graph based on a binomial method in theoretical report
    """
    def __init__(self, ni, nd, no, depth_max, p0, w0, l0, tau, capacity, basis='canonical', ax_p=None):

        # Core feature of graph
        self.ni, self.nd, self.no, self.depth_max, self.p0, self.w0 = ni, nd, no, depth_max, p0, w0

        # Core feature of nodes
        self.l0, self.tau, self.basis, self.capacity = l0, tau, basis, capacity
        self.l_edges = []

        # Init binomial construction
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
        self.l_edges, self.nd = self.reindex_nodes(l_edges, self.gammas)

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

    def reindex_nodes(self, l_edges, gammas):
        """

        :param l_edges:
        :param gammas:
        :return:
        """
        # Gather nodes
        l_nodes = sum(gammas[1:], [])

        # Build index
        d_index = dict([(k, 'network_{}'.format(i)) for i, k in enumerate(l_nodes)])

        # Transform edges and network nodes
        l_edges_ = [(nx, d_index[ny]) if 'input' in nx else (d_index[nx], d_index[ny]) for nx, ny in l_edges]
        nd = len(d_index.values())

        return l_edges_, nd

    def build_matrices(self):
        """

        :return:
        """
        if len(self.l_edges) == 0:
            self.build_tuples()

        return mat_from_tuples(self.l_edges, self.ni, self.nd, self.no, weights=[self.w0] * len(self.l_edges))

    def build_network(self, project, network_id=None):
        """

        :return:
        """

        if len(self.l_edges) == 0:
            self.build_tuples()

        mat_in, mat_net, mat_out = self.build_matrices()

        return DeepNetwork.from_matrices(project, mat_net, mat_in, mat_out, self.capacity, self.basis, w0=self.w0,
                                         l0=self.l0, tau=self.tau, network_id=network_id)

    @staticmethod
    def layout(sax_I, sax_D, sax_O):

        # Get dimension of graph
        ni, nd, no = sax_I.shape[0], sax_D.shape[0], sax_O.shape[-1]

        # Compute network node's positions
        end, ax_sn, ax_si, ax_pos, i = False, np.zeros(nd), np.ones(ni), np.zeros(nd), 0

        while not end:
            if i > 0:
                ax_si = np.zeros(ni)
            ax_sn = ax_sn.dot(sax_D.toarray()) + ax_si.dot(sax_I.toarray())
            ax_pos += (i * ((ax_pos == 0) & np.array(ax_sn > 0))) + 1
            if (ax_sn == 0).all() or (ax_pos > 0).all():
                end = True

            i += 1

        pos = {'inputs': {'pos': {i: (0, (nd - ni) / 2 + i) for i in range(ni)}, 'color': 'r'}}
        pos.update({'networks': {'pos': {ni + i: (ax_pos[i] + (np.random.randn() / 5), np.random.randint(0, nd))
                                         for i in range(nd)}, 'color': 'k'}})
        pos.update({'outputs': {'pos': {ni + nd + i: (max(ax_pos) + 1, (nd - no) / 2 + i) for i in range(no)}, 'color': 'b'}})

        return pos

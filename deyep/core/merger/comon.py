# Global import
import numpy as np
from scipy.sparse import csc_matrix, hstack

# Local import
from deyep.core.datastructures.deep_network import DeepNetwork


class DeepNetMerger(object):
    def __init__(self, l_deep_networks, basis):

        # Core params
        self.basis = basis
        self.l_deep_networks = l_deep_networks
        self.deep_network = None

    def merge_network(self):

        if len(self.l_deep_networks) == 1:
            self.deep_network = self.deep_network[0]
            return self

        else:
            raise NotImplementedError

    def clean_network(self):
        if self.deep_network is None:
            self.merge_network()

        sax_I, sax_D, sax_O = self.deep_network.graph('I'), self.deep_network.graph('D'), self.deep_network.graph('O')

        d_layers = self.classify_nodes_by_layer(sax_I, sax_D, self.deep_network.network_nodes)

        for _, l_nodes in sorted(d_layers.items(), key=lambda x: x[0]):
            sax_O, sax_D = self.matrix_cleaning(l_nodes, sax_I, sax_D, sax_O)

        ax_active_nodes = sax_O.sum(axis=1) > 0
        ax_active_nodes &= sax_I.sum(axis=1) > 0

        DeepNetwork.reduce_network(self.deep_network, ax_active_nodes, self.basis)

        return self

    def classify_nodes_by_layer(self, sax_I, sax_D, l_nodes):

        n, delta, sax_layers = à, 1, csc_matrix(np.ones((sax_I.shape[0], 1))).dot(sax_I) > 0

        while delta > 0:
            sax_s = sax_layers.dot(sax_D) > 0
            sax_layers += sax_s
            delta = sax_s.sum()
            n += 1

        return {i + 1: [(j, l_nodes[j]) for j in (sax_layers == i + 1).nonzeros()[-1]] for i in range(n)}

    def matrix_cleaning(self, l_nodes, sax_I, sax_D, sax_O):

        if self.deep_network is None:
            self.merge_network()

        changed = True

        while changed:
            changed = False
            for i, na in l_nodes:
                # Get input map for node nb
                sax_in_na = hstack(sax_I[:, i].transpose(), sax_D[:, i].transpose())

                # get level representation
                l_max_na = max(sax_D[:, i].sum() + sax_I[:, i].sum(), sax_D[:, j].sum() + sax_I[:, j].sum())
                sax_l_na = csc_matrix([na.d_levels[k + 1] for k in range(l_max_na)])
                if sax_l_na.nnz == 0 and sax_D[i, :].nnz > 0:
                    sax_D[i, :] = 0
                    sax_O[i, :] = 0
                    changed = True
                    continue

                for j, nb in l_nodes:
                    if i == j:
                        continue
                    # Get input map for node nb
                    sax_in_nb = hstack(sax_I[:, j].transpose(), sax_D[:, j].transpose())

                    # get level representation
                    l_max_nb = sax_D[:, j].sum() + sax_I[:, j].sum()
                    sax_l_nb = csc_matrix([nb.d_levels[k + 1] for k in range(l_max_nb)])

                    # Check if any of the nodes a no
                    if sax_l_nb.nnz == 0 and sax_D[j, :].nnz > 0:
                        sax_D[j, :] = 0
                        sax_O[j, :] = 0
                        changed = True
                        continue

                    # Get product
                    v = sax_in_na.dot(sax_in_nb)[0, 0] + \
                        sax_l_na[:, min(l_max_nb, l_max_na)].dot(sax_l_nb[:, min(l_max_nb, l_max_na)])[0, 0]

                    # Case exact match between incoming edge of na and nb
                    if v == (sax_in_na.sum() + sax_l_na.sum()) and v == (sax_in_nb.sum() + sax_l_nb.sum()):
                        if sax_D[j, :].nnz > 0:
                            sax_D[i, :] += sax_D[j, :]
                            sax_D[j, :] = 0
                            sax_O[i, :] += sax_O[j, :]
                            sax_O[j, :] = 0
                            changed = True
        return sax_D, sax_O

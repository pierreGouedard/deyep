# Global import
import numpy as np
from scipy.sparse import csc_matrix, hstack, vstack
import random
import string

# Local import
from deyep.core.datastructures.deep_network import DeepNetwork
from deyep.core.datastructures.deep_network_dry import DeepNetworkDry


class DeepNetMerger(object):
    def __init__(self, l_deep_networks):

        # Core params
        self.l_deep_networks = l_deep_networks
        self.deep_network = None

        if len(self.l_deep_networks) == 1:
            self.deep_network = self.l_deep_networks[0]

    def merge_network(self):

        # If dried network use simplified procedure
        if all([dn.is_dried for dn in self.l_deep_networks]):
            return self.merge_network_dry()

        # Init deep network
        self.deep_network = self.l_deep_networks[0]

        # Merge network iteratively
        for dn in self.l_deep_networks[1:]:
            if dn.is_dried:
                raise ValueError('Could not merge dry and non dry network together')

            sax_I, sax_O, sax_D, sax_Cm, levels = DeepNetMerger.merge_matrices(self.deep_network, dn, is_dried=False)
            DeepNetMerger.check_comptability(self.deep_network, dn)

            self.deep_network = DeepNetwork.from_matrices(
                self.deep_network.project, sax_D, sax_I, sax_O, dn.node_capacity, basis=dn.node_basis, w0=dn.w0,
                l0=dn.l0, tau=dn.tau, Cm=sax_Cm, levels=levels, network_id='network_{}'.format(self.rid(7))
            )

            self.clean_network()

        return self

    @staticmethod
    def rid(n):
            return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

    def merge_network_dry(self):

        # Init deep network
        self.deep_network = self.l_deep_networks[0]

        # Merge network iteratively
        for dn in self.l_deep_networks[1:]:
            sax_I, sax_O, sax_D, levels = DeepNetMerger.merge_matrices(self.deep_network, dn, is_dried=True)

            self.deep_network = DeepNetworkDry.from_matrices(self.deep_network.project, sax_D, sax_I, sax_O,
                                                             levels=levels)
            self.clean_network()

        return self

    def clean_network(self):
        if self.deep_network is None:
            return self.merge_network()

        tau = self.deep_network.tau if not self.deep_network.is_dried else None
        sax_I, sax_D, sax_O = self.deep_network.I, self.deep_network.D, self.deep_network.O

        # Forward cleaning
        sax_D, sax_O, d_active = self.clean_network_(sax_I, sax_D, sax_O, tau=tau)

        # Backward cleaning
        sax_D, sax_I, d_active = self.clean_network_(sax_O.transpose(), sax_D.transpose(), sax_I.transpose(),
                                                     d_active=d_active)

        # Compute active node mask
        ax_active_nodes = np.array([d_active.get(i, True) for i in range(sax_D.shape[0])], dtype=bool)

        # Update structure of deep network
        self.deep_network.graph['Dw'] = self.deep_network.graph['Dw'].multiply(self.deep_network.D).data.mean() * sax_D.transpose()
        self.deep_network.graph['Iw'] = self.deep_network.graph['Iw'].multiply(self.deep_network.I).data.mean() * sax_I.transpose()
        self.deep_network.graph['Ow'] = self.deep_network.graph['Ow'].multiply(self.deep_network.O).data.mean() * sax_O

        if self.deep_network.is_dried:
            self.deep_network = DeepNetworkDry.reduce_network(self.deep_network, ax_active_nodes)
        else:
            self.deep_network = DeepNetwork.reduce_network(self.deep_network, ax_active_nodes)

        return self

    def clean_network_(self, sax_I, sax_D, sax_O, d_active=None, tau=None):

        # Get nodes classified by layers (layer 0 are nodes without incoming
        d_layers, d_active = self.classify_nodes_by_layer(sax_I, sax_D, self.deep_network.network_nodes, d_active)

        for _, l_nodes in sorted(d_layers.items(), key=lambda x: x[0]):
            # Merge nodes of same layer
            sax_D, sax_O, d_active = self.matrix_cleaning(l_nodes, sax_I, sax_D, sax_O, d_active, tau=tau)

        return sax_D, sax_O, d_active

    @staticmethod
    def classify_nodes_by_layer(sax_I, sax_D, l_nodes, d_active=None):

        n, delta, sax_layers = 1, 1, (csc_matrix(np.ones((1, sax_I.shape[0]))).dot(sax_I) > 0).astype(int)
        sax_s = sax_layers.astype(bool).copy()

        while delta > 0:
            sax_s = sax_s.dot(sax_D) > 0
            sax_layers[sax_s] = n + 1
            delta = sax_s.sum()
            n += 1

        # Get active nodes
        if d_active is not None:
            d_active.update({i: False for i in (sax_layers == 0).nonzero()[-1]})
        else:
            d_active = {i: False for i in (sax_layers == 0).nonzero()[-1]}

        # Get node by layers
        d_nodes = {i: [(j, l_nodes[j]) for j in (sax_layers == i).nonzero()[-1]] for i in range(n)}

        return d_nodes, d_active

    @staticmethod
    def merge_matrices(dna, dnb, is_dried=False):
        levels = [n.d_levels for n in dna.network_nodes]
        levels += [n.d_levels for n in dnb.network_nodes]

        if is_dried:
            sax_I, sax_O = hstack([dna.I, dnb.I], format='csc'), vstack([dna.O, dnb.O], format='csc')
            sax_D = vstack([hstack([dna.D, csc_matrix((dna.D.shape[0], dnb.D.shape[1]))]),
                            hstack([csc_matrix((dnb.D.shape[0], dna.D.shape[1])), dnb.D])], format='csc')

            return sax_I, sax_O, sax_D, levels

        else:
            sax_I = hstack([dna.Iw.multiply(dna.I), dnb.Iw.multiply(dnb.I)], format='csc')
            sax_O = vstack([dna.Ow.multiply(dna.O), dnb.Ow.multiply(dnb.O)], format='csc')
            sax_Cm = vstack([dna.Cm, dnb.Cm], format='csc')

            sax_D = vstack([
                hstack([dna.Dw.multiply(dna.D), csc_matrix((dna.D.shape[0], dnb.D.shape[1]))], format='csc'),
                hstack([csc_matrix((dnb.D.shape[0], dna.D.shape[1])), dnb.Dw.multiply(dnb.D)], format='csc'),
            ], format='csc')

            return sax_I, sax_O, sax_D, sax_Cm, levels

    @staticmethod
    def check_comptability(dna, dnb):
        try:
            assert dna.w0 == dnb.w0
            assert dna.l0 == dnb.l0
            assert dna.tau == dnb.tau
            assert dna.project == dnb.project
        except AssertionError:
            raise ValueError('networks merged should have same parameters w0, l0, tau and project')

        try:
            assert dna.node_capacity == dnb.node_capacity
        except AssertionError:
            raise ValueError('nodes of merged network should have same parameters n_freq, node_capacity and node_basis')

    @staticmethod
    def matrix_cleaning(l_nodes, sax_I, sax_D, sax_O, d_active, tau=None):

        for n, (i, na) in enumerate(l_nodes):

            if not d_active.get(na.id, True):
                sax_D[i, :], sax_O[i, :] = False, False
                continue

            # Get input map for node nb
            sax_in_na = hstack([sax_I[:, i].transpose(), sax_D[:, i].transpose()]).astype(int)

            # If no input, then deactivate node
            if sax_in_na.sum() == 0:
                sax_D[i, :], sax_O[i, :], d_active[na.id] = False, False, False
                continue

            # get level representation
            l_max_na = sax_D[:, i].sum() + sax_I[:, i].sum()
            if tau is not None:
                sax_l_na = csc_matrix([na.d_levels[k + 1] > tau for k in range(l_max_na)]).astype(int)
            else:
                sax_l_na = csc_matrix([na.d_levels.get(k + 1, False) for k in range(l_max_na)]).astype(int)

            # Check if at east one level activate
            if sax_l_na.nnz == 0:
                sax_D[i, :], sax_O[i, :], d_active[na.id] = False, False, False
                continue

        sax_D.eliminate_zeros()
        sax_O.eliminate_zeros()

        return sax_D, sax_O, d_active


#### OLD CODE FOR MERGE
    #
    # if not merge_duplicated:
    #     continue
    # print 'here mother fucker'
    # for j, nb in l_nodes[n + 1:]:
    #
    #     if not d_active.get(nb.id, True):
    #         sax_D[j, :], sax_O[j, :] = False, False
    #         continue
    #
    #     # Get input map for node nb
    #     sax_in_nb = hstack([sax_I[:, j].transpose(), sax_D[:, j].transpose()]).astype(int)
    #
    #     # If no input, then deactivate node
    #     if sax_in_nb.sum() == 0:
    #         sax_D[j, :], sax_O[j, :], d_active[nb.id] = False, False, False
    #         continue
    #
    #     # get level representation
    #     l_max_nb = sax_D[:, j].sum() + sax_I[:, j].sum()
    #     if tau is not None:
    #         sax_l_nb = csc_matrix([nb.d_levels[k + 1] > tau for k in range(l_max_nb)]).astype(int)
    #     else:
    #         sax_l_nb = csc_matrix([nb.d_levels.get(k + 1, False) for k in range(l_max_nb)]).astype(int)
    #
    #     # Check if at east one level activate
    #     if sax_l_nb.nnz == 0:
    #         sax_D[j, :], sax_O[j, :], d_active[nb.id] = False, False, False
    #         continue
    #
    #     if l_max_nb == l_max_na:
    #         # Get product
    #         v = sax_in_na.dot(sax_in_nb.transpose())[0, 0] + sax_l_na.dot(sax_l_nb.transpose())[0, 0]
    #
    #         # Case exact match between incoming edge of na and nb
    #         if v == (sax_in_na.sum() + sax_l_na.sum()) and v == (sax_in_nb.sum() + sax_l_nb.sum()):
    #             # merge nodes outputs
    #             sax_D[i, :] += sax_D[j, :]
    #             sax_O[i, :] += sax_O[j, :]
    #
    #             # Deactivate one of the clone node
    #             sax_D[j, :], sax_O[j, :], d_active[nb.id] = False, False, False
    #
    #             # Get common children to indicate level update
    #             l_children = list({t[0] for t in na.children}.intersection({t[0] for t in nb.children}))
    #             d_levels.update(
    #                 {k: d_levels.get(k, 0) + 1 for k in l_children}
    #             )

    @staticmethod
    def update_levels(l_nodes, d_levels):

        for k, n in d_levels.items():
            # Get node of interest
            node = l_nodes[int(k.split('_')[1])]

            # Update level
            node.d_levels.update({k_ - n: v_ for k_, v_ in node.d_levels.items() if k_ > n})

            # Replace old node
            l_nodes[int(k.split('_')[1])] = node

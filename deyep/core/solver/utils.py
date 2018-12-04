# Global import
import numpy as np
from scipy.sparse import csc_matrix, hstack, vstack
import random
import string

# Local import
from deyep.core.datastructures.deep_network import DeepNetwork
from deyep.core.datastructures.deep_network_dry import DeepNetworkDry


class DeepNetUtils(object):
    def __init__(self, deep_network):

        # Core params
        self.deep_network = deep_network
        self.mapping_main_sub = None

    @staticmethod
    def rid(n):
            return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

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
    def cluster(sax_I, sax_D, l_nodes, d_active=None):

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

    def merge(self, l_dn):

        # If dried network use simplified procedure
        if self.deep_network.is_dried:
            return self.merge_dry(l_dn)

        # Merge network iteratively
        for dn in l_dn:
            if dn.is_dried:
                raise ValueError('Could not merge dry and non dry network together')

            sax_I, sax_O, sax_D, sax_Cm, levels = DeepNetUtils.merge_matrices(self.deep_network, dn, is_dried=False)
            DeepNetUtils.check_comptability(self.deep_network, dn)

            self.deep_network = DeepNetwork.from_matrices(
                self.deep_network.project, sax_D, sax_I, sax_O, dn.node_capacity, w0=dn.w0,
                l0=dn.l0, tau=dn.tau, Cm=sax_Cm, levels=levels, network_id='network_{}'.format(self.rid(7))
            )

            self.clean_multi_side()

        return self

    def merge_dry(self, l_dn):

        # Merge network iteratively
        for dn in l_dn:
            if not dn.is_dried:
                raise ValueError('Could not merge dry and non dry network together')

            sax_I, sax_O, sax_D, levels = DeepNetUtils.merge_matrices(self.deep_network, dn, is_dried=True)

            self.deep_network = DeepNetworkDry.from_matrices(
                self.deep_network.project, sax_D, sax_I, sax_O, levels=levels
            )
            self.clean_multi_side()

        return self

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

    def combine(self, main=None, sub=None, d_map=None, binary=True):
        if (main is None and sub is None) or d_map is None:
            raise ValueError('One of "main" or "sub" args shuld be filled. d_map should be filled too')

        if main is not None:
            sub = self.deep_network.copy()
        else:
            main = self.deep_network.copy()

        # Get core data strucure (untouched nodes)
        l_main = [nid for nid in range(main.D.shape[0]) if nid not in d_map.keys()]
        d_main = self.compute_main_matrices(main, l_main)

        # Get first level configuration of additional network
        d_sub_odd = self.compute_sub_matrices(
            main, sub, {id_: l_d[0] for id_, l_d in d_map.items() if len(l_d) > 0 and isinstance(id_, int)}, l_main,
            d_map.get('ido', range(len(sub.network_nodes)))
        )

        if binary:
            # Get second level configuration of additional network
            d_sub_even = self.compute_sub_matrices(
                main, sub, {id_: l_d[1] for id_, l_d in d_map.items() if len(l_d) > 1 and isinstance(id_, int)}, l_main,
                d_map.get('ide', range(len(sub.network_nodes))), even=True
            )

            # Combine all matrices and levels
            sax_I, sax_D, sax_O, sax_Cm, levels, mapping = self.combine_matrices(d_main, d_sub_odd, d_sub_e=d_sub_even)

        else:
            raise NotImplementedError

        self.deep_network = DeepNetwork.from_matrices(
            main.project, sax_D, sax_I, sax_O, main.node_capacity, w0=main.w0,
            l0=main.l0, tau=main.tau, Cm=sax_Cm, levels=levels, network_id=main.network_id
        )
        self.mapping_main_sub = mapping
        return self

    @staticmethod
    def compute_main_matrices(dn, l_main):

        # Reduce matrices
        sax_I = dn.Iw.multiply(dn.I)[:, l_main]
        sax_D = dn.Dw.multiply(dn.D)[l_main, :][:, l_main]
        sax_O = dn.Ow.multiply(dn.O)[l_main, :]
        sax_Cm = dn.Cm[l_main, :]

        # Update levels
        levels = {n: dn.network_nodes[i].d_levels for n, i in enumerate(l_main)}

        return {'I': sax_I, 'D': sax_D, 'O': sax_O, 'Cm': sax_Cm, 'l': levels}

    @staticmethod
    def compute_sub_matrices(dnm, dns, d_map, l_main, l_sub, even=False):

        d_struct, l_out = {}, sorted(d_map.keys())
        sax_Im, sax_Dm, sax_Om, sax_Cm = dnm.Iw.multiply(dnm.I), dnm.Dw.multiply(dnm.D), dnm.Ow.multiply(dnm.O), dnm.Cm
        sax_Is, sax_Ds, sax_Os = dns.Iw.multiply(dns.I), dns.Dw.multiply(dns.D), dns.Ow.multiply(dns.O)

        # Combine Input matrices
        if dns.IOw is not None:
            sax_Is = csc_matrix((sax_Im.shape[0], 0))
            for nid, d in d_map.items():
                sax_Im[:, nid] += dns.IOw[:, d['id']]

        if even:
            d_struct['I'] = hstack([sax_Im[:, l_out], sax_Is[:, l_sub]], format='csc')
        else:
            d_struct['I'] = hstack([sax_Is[:, l_sub], sax_Im[:, l_out]], format='csc')

        # Combine upper network matrices
        d_struct['Dmi'], d_struct['Dmo'] = sax_Dm[l_main, :][:, l_out], sax_Dm[l_out, :][:, l_main]

        # Combine lower network matrices
        d_struct['Ds'] = sax_Ds[l_sub, :][:, l_sub]
        d_struct['Os'] = csc_matrix((d_struct['Ds'].shape[0], len(l_out)))

        if sax_Os.nnz > 0:
            for i, (_, d) in enumerate(sorted(d_map.items(), key=lambda x: x[0])):
                if sax_Os[l_sub, d['id']].nnz > 0:
                    d_struct['Os'][:, i] = sax_Os[l_sub, d['id']]

        # Combine output and candidate matrices
        if even:
            d_struct['O'] = vstack([sax_Om[l_out, :], csc_matrix((len(l_sub), sax_Om.shape[1]))], format='csc')
            d_struct['Cm'] = vstack([sax_Cm[l_out, :], csc_matrix((len(l_sub), sax_Om.shape[1]))], format='csc')\
                .astype(bool)
        else:
            d_struct['O'] = vstack([csc_matrix((len(l_sub), sax_Om.shape[1])), sax_Om[l_out, :]], format='csc')
            d_struct['Cm'] = vstack([csc_matrix((len(l_sub), sax_Om.shape[1])), sax_Cm[l_out, :]], format='csc')\
                .astype(bool)

        # Level update
        d_struct['l_out'] = {i: d['level'] for i, (_, d) in enumerate(sorted(d_map.items(), key=lambda x: x[0]))}
        d_struct['l_in'] = {l_sub.index(n.id): n.d_levels for n in dns.network_nodes if n.id in l_sub}
        d_struct['ids'] = [n.id for n in dns.network_nodes if n.id in l_sub]
        return d_struct

    @staticmethod
    def combine_matrices(d_main, d_sub_o, d_sub_e=None):

        if d_sub_e is None:
            raise NotImplementedError

        # Combine input, output and candidate memory matrices
        sax_I = hstack([d_sub_o['I'], d_main['I'], d_sub_e['I']], format='csc')
        sax_O = vstack([d_sub_o['O'], d_main['O'], d_sub_e['O']], format='csc')
        sax_Cm = vstack([d_sub_o['Cm'], d_main['Cm'], d_sub_e['Cm']], format='csc')

        # Second and fourth line of network matrix
        padl, padr = d_sub_o['Ds'].shape[1] + d_sub_o['Os'].shape[1], d_sub_e['Ds'].shape[1] + d_sub_e['Os'].shape[1]
        sax_D2 = hstack(
            [csc_matrix((d_sub_o['Dmo'].shape[0], padl)), d_sub_o['Dmo'], csc_matrix((d_sub_o['Dmo'].shape[0], padr))],
            format='csc'
        )
        sax_D4 = hstack(
            [csc_matrix((d_sub_e['Dmo'].shape[0], padl)), d_sub_e['Dmo'], csc_matrix((d_sub_e['Dmo'].shape[0], padr))],
            format='csc'
        )

        # First and fifth line of netword matrix
        padl, padr = d_main['D'].shape[1] + padl, d_main['D'].shape[1] + padr
        sax_D1 = hstack(
            [d_sub_o['Ds'], d_sub_o['Os'], csc_matrix((d_sub_o['Ds'].shape[0], padr))],
            format='csc'
        )
        sax_D5 = hstack(
            [csc_matrix((d_sub_e['Ds'].shape[0], padl)), d_sub_e['Os'], d_sub_e['Ds']],
            format='csc'
        )

        # Middle line of network matrix
        padl, padr = d_sub_o['Ds'].shape[1], d_sub_e['Ds'].shape[1]
        sax_D3 = hstack(
            [csc_matrix((d_main['D'].shape[0], padl)), d_sub_o['Dmi'], d_main['D'], d_sub_e['Dmi'],
             csc_matrix((d_main['D'].shape[0], padr))],
            format='csc'
        )

        # Gather rows of network matrix
        sax_D = vstack([sax_D1, sax_D2, sax_D3, sax_D4, sax_D5], format='csc')

        # Get levels
        levels, n = {}, 0
        for d_level in [d_sub_o['l_in'], d_sub_o['l_out'], d_main['l'], d_sub_e['l_out'], d_sub_e['l_in']]:
            levels.update({k + n: v for k, v in d_level.items()})
            if len(levels) > 0:
                n = max(levels.keys()) + 1

        # Get mapping main (network nodes) -> sub (network nodes)
        mapping = {i: nid for i, nid in d_sub_o['ids']}
        mapping.update({len(levels) - len(d_sub_e['l_in']) + i: nid for i, nid in d_sub_e['ids']})

        return sax_I, sax_D, sax_O, sax_Cm, levels, mapping

    @staticmethod
    def transform_level(level, l0):
        if isinstance(level.values()[0], bool) or isinstance(level.values()[0], np.bool_):
            return {k: l0 if v else 0 for k, v in level.items()}
        else:
            return level

    def clean_multi_side(self):

        tau = self.deep_network.tau if not self.deep_network.is_dried else None
        sax_I, sax_D, sax_O = self.deep_network.I, self.deep_network.D, self.deep_network.O

        # Forward cleaning
        sax_D, sax_O, d_active = self.clean(sax_I, sax_D, sax_O, tau=tau)

        # Backward cleaning
        sax_D, sax_I, d_active = self.clean(sax_O.transpose(), sax_D.transpose(), sax_I.transpose(), d_active=d_active)

        # Compute active node mask
        ax_active_nodes = np.array([d_active.get(i, True) for i in range(sax_D.shape[0])], dtype=bool)

        # Update structure of deep network
        self.deep_network.graph['Dw'] = self.deep_network.graph['Dw'].multiply(sax_D.transpose())
        self.deep_network.graph['Iw'] = self.deep_network.graph['Iw'].multiply(sax_I.transpose())
        self.deep_network.graph['Ow'] = self.deep_network.graph['Ow'].multiply(sax_O)

        if self.deep_network.is_dried:
            self.deep_network = DeepNetworkDry.reduce_network(self.deep_network, ax_active_nodes)
        else:
            self.deep_network = DeepNetwork.reduce_network(self.deep_network, ax_active_nodes)

        return self

    def clean(self, sax_I, sax_D, sax_O, d_active=None, tau=None):

        # Get nodes classified by layers (layer 0 are nodes without incoming
        d_layers, d_active = self.cluster(sax_I, sax_D, self.deep_network.network_nodes, d_active)

        for _, l_nodes in sorted(d_layers.items(), key=lambda x: x[0]):
            # Merge nodes of same layer
            sax_D, sax_O, d_active = self.clean_matrix(l_nodes, sax_I, sax_D, sax_O, d_active, tau=tau)

        return sax_D, sax_O, d_active

    @staticmethod
    def clean_matrix(l_nodes, sax_I, sax_D, sax_O, d_active, tau=None):

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

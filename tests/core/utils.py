# Global imports
import unittest
from scipy.sparse import csc_matrix as csc, hstack, vstack
import numpy as np

# Local import
from deyep.core.datastructures.deep_network import DeepNetwork
from deyep.core.builder.binomial import BinomialGraphBuilder
from deyep.core.solver.utils import DeepNetUtils

__maintainer__ = 'Pierre Gouedard'


class Testutils(unittest.TestCase):
    def setUp(self):

        self.params_network = {'ni': 10, 'no': 10, 'nd': 50, 'p0': 0.05, 'w0': 5, 'l0': 10, 'tau': 5, 'c': 10}
        self.builder = BinomialGraphBuilder
        np.random.seed(1234)

    def test_merge(self):
        raise NotImplementedError

    def test_clean(self):
        raise NotImplementedError

    def combine_case_1(self):
        """
        Combine general case (depth of network > 1)
        python -m unittest tests.core.utils.Testutils.combine_case_1

        """
        dn_main, dn_sub, d_mapping, dn_combined = set_up_combine_case_1(self.params_network, self.builder)

        dn_util = DeepNetUtils(dn_sub)
        dn_combined_ = dn_util.combine(main=dn_main, d_map=d_mapping).deep_network

        # Check combined network are equal
        self.assertTrue((dn_combined_.I.toarray().astype(int) == dn_combined.I.toarray().astype(int)).all())
        self.assertTrue((dn_combined_.D.toarray().astype(int) == dn_combined.D.toarray().astype(int)).all())
        self.assertTrue((dn_combined_.O.toarray().astype(int) == dn_combined.O.toarray().astype(int)).all())

        # Make sure combiner node's levels are correct
        offset_o, offset_e, l_level_checked = len(d_mapping.pop('ido')), len(d_mapping.pop('ide')), []

        # Odd combiner
        l_levels = [l_d[0]['level'] for _, l_d in sorted(d_mapping.items(), key=lambda x: x[0])]
        for i, l in enumerate(l_levels):
            self.assertEqual(dn_combined_.network_nodes[offset_o + i].d_levels, l)

        # Even combiner
        l_levels = [l_d[1]['level'] for _, l_d in sorted(d_mapping.items(), key=lambda x: x[0]) if len(l_d) > 1]
        for i, l in enumerate(l_levels):
            self.assertEqual(dn_combined_.network_nodes[i - offset_e - len(l_levels)].d_levels, l)

    def combine_case_2(self):
        """
        Combine specific case (depth of network == 1)
        python -m unittest tests.core.utils.Testutils.combine_case_2

        """
        dn_main, dn_sub, d_mapping, dn_combined = set_up_combine_case_2(self.params_network, self.builder)

        # Combine Network
        dn_util = DeepNetUtils(dn_sub)
        dn_combined_ = dn_util.combine(main=dn_main, d_map=d_mapping).deep_network

        # Check combined network are equal
        self.assertTrue((dn_combined_.I.toarray().astype(int) == dn_combined.I.toarray().astype(int)).all())
        self.assertTrue((dn_combined_.D.toarray().astype(int) == dn_combined.D.toarray().astype(int)).all())
        self.assertTrue((dn_combined_.O.toarray().astype(int) == dn_combined.O.toarray().astype(int)).all())

        # Make sure combiner node's levels are correct
        # Odd combiner
        l_levels = [l_d[0]['level'] for _, l_d in sorted(d_mapping.items(), key=lambda x: x[0])]
        for i, l in enumerate(l_levels):
            self.assertEqual(dn_combined_.network_nodes[i].d_levels, l)

        # Even combiner
        l_levels = [l_d[1]['level'] for _, l_d in sorted(d_mapping.items(), key=lambda x: x[0]) if len(l_d) > 1]
        for i, l in enumerate(l_levels):
            self.assertEqual(dn_combined_.network_nodes[i - len(l_levels)].d_levels, l)


def set_up_combine_case_1(pn, builder):
    """
    Visual inspection
    ax_graph = gather_matrices(dn.Iw.toarray(), dn.Dw.toarray(), dn.Ow.toarray())
    plot_graph(ax_graph, builder.layout(dn.Iw.toarray(), dn.Dw.toarray(), dn.Ow.toarray()))

    """

    # Build lower and upper core
    nu, nl, dnu, dnl, D_lu = build_core(builder, pn)

    # Build combiner nodes and sub network
    no, ne, ns, I_o, D_ou, D_lo, O_o = build_combiner(nu, nl, pn)

    # Build sub network and mapping
    nso, nse, dnso, dnse, d_mapping = build_sub(builder, pn, nu, ns, no, ne)

    # Build networks
    dn_main = build_main_network(pn, nu, no, nl, dnu, dnl, D_lu, I_o, D_ou, D_lo, O_o)
    dn_sub, d_mapping = build_sub_network(pn, no, ne, nso, nse, dnso, dnse, d_mapping)
    dn_merged = build_merged_network(pn, nu, nl, no, ne, nso, nse, dnu, dnl, D_lu, dnso, dnse, I_o, D_ou, D_lo, O_o)

    return dn_main, dn_sub, d_mapping, dn_merged


def set_up_combine_case_2(pn, builder):
    """
    Visual inspection
    ax_graph = gather_matrices(dn.Iw.toarray(), dn.Dw.toarray(), dn.Ow.toarray())
    plot_graph(ax_graph, builder.layout(dn.Iw.toarray(), dn.Dw.toarray(), dn.Ow.toarray()))

    """

    # Build lower and upper core
    nm, dnm = build_core_2(builder, pn)

    # Build combiner nodes and sub network
    no, ne, I_o, D_om, O_o = build_combiner_2(nm, pn)

    # Build sub network and mapping
    dn_sub, d_mapping = build_sub_2(pn, nm, no, ne)

    # Build networks
    dn_main = build_main_network_2(pn, nm, no, dnm, I_o, D_om, O_o)
    dn_merged = build_merged_network_2(pn, nm, no, ne, dnm, dn_sub, I_o, D_om, O_o, d_mapping)

    return dn_main, dn_sub, d_mapping, dn_merged


def build_core(builder, pn):
    nu, nl = int(pn['nd'] * 0.2), int(pn['nd'] * 0.3)
    dnu = builder(pn['ni'], nu, pn['no'], 2, pn['p0'], pn['w0'], pn['l0'], pn['tau'], pn['c'])\
        .build_network('core_upper')
    dnl = builder(pn['ni'], nl, pn['no'], 3, pn['p0'], pn['w0'], pn['l0'], pn['tau'], pn['c'])\
        .build_network('core_lower')
    D_lu = rs([len(dnl.network_nodes), len(dnu.network_nodes)], w0=pn['w0'])
    dnu.graph['Ow'] = rs([len(dnu.network_nodes), pn['no']], w0=pn['w0'])

    return len(dnu.network_nodes), len(dnl.network_nodes), dnu, dnl, D_lu


def build_core_2(builder, pn):
    nm = int(pn['nd'] * 0.9)
    dnm = builder(pn['ni'], nm, pn['no'], 3, pn['p0'], pn['w0'], pn['l0'], pn['tau'], pn['c'])\
        .build_network('main')

    dnm.graph['Ow'] = rs([len(dnm.network_nodes), pn['no']], w0=pn['w0'])

    return len(dnm.network_nodes), dnm


def build_combiner(nu, nl, pn):

    no, ne, ns = int(pn['nd'] * 0.06), int(pn['nd'] * 0.04), int(pn['nd'] * 0.2)
    I_o, D_ou, D_lo, O_o = rs([pn['ni'], no], w0=pn['w0']), rs([no, nu], w0=pn['w0']), rs([nl, no], w0=pn['w0']), \
        rs([no, pn['no']], w0=pn['w0'])

    return no, ne, ns, I_o, D_ou, D_lo, O_o


def build_combiner_2(nm, pn):

    no, ne = int(pn['nd'] * 0.06), int(pn['nd'] * 0.04)
    I_o, D_om, O_o = rs([pn['ni'], no], w0=pn['w0']), rs([no, nm], w0=pn['w0']), rs([no, pn['no']], w0=pn['w0'])
    return no, ne, I_o, D_om, O_o


def build_sub(builder, pn, nu, ns, no, ne):
    dnso = builder(pn['ni'], ns, no, 3, pn['p0'], pn['w0'], pn['l0'], pn['tau'], pn['c'])\
        .build_network('sub_odd')
    dnse = builder(pn['ni'], ns, ne, 3, pn['p0'], pn['w0'], pn['l0'], pn['tau'], pn['c'])\
        .build_network('sub_even')

    nso, nse = len(dnso.network_nodes), len(dnse.network_nodes)
    dnso.graph['Ow'], dnse.graph['Ow'] = rs([nso, no], pn['w0']), rs([nse, ne], pn['w0'])

    # Build mapping
    d_mapping = {
        nu + i: [{'id': 2 * i, 'level': rl(100, pn['l0'])}, {'id': 2 * i + 1, 'level': rl(100, pn['l0'])}]
        for i in range(ne)
    }
    d_mapping.update({
        max(d_mapping.keys()) + 1 + i: [{'id': 2 * ne + i, 'level': rl(100, pn['l0'])}] for i in range(no - ne)
    })

    return nso, nse, dnso, dnse, d_mapping


def build_sub_2(pn, nm, no, ne):

    graph = {'Iw': csc((0, 0)), 'Dw': csc((0, 0)), 'Ow': csc((0, 0)), 'Cm': csc((0, 0)),
             'IOw': rs([pn['ni'], no + ne])}
    dns = DeepNetwork('sub', pn['w0'], pn['l0'], pn['tau'], [], [], [], graph, 0, 0)

    # Build mapping
    d_mapping = {
        nm + i: [{'id': 2 * i, 'level': rl(100, pn['l0'])}, {'id': 2 * i + 1, 'level': rl(100, pn['l0'])}]
        for i in range(ne)
    }
    d_mapping.update({
        max(d_mapping.keys()) + 1 + i: [{'id': 2 * ne + i, 'level': rl(100, pn['l0'])}] for i in range(no - ne)
    })

    return dns, d_mapping


def build_main_network(pn, nu, no, nl, dnu, dnl, D_lu, I_o, D_ou, D_lo, O_o):
    # Build Main matrices:
    sax_I = hstack([dnu.Iw, I_o, dnl.Iw], format='csc')

    sax_D = vstack([
        hstack([dnu.Dw, csc((nu, no)), csc((nu, nl))]),
        hstack([D_ou, csc((no, no)), csc((no, nl))]),
        hstack([D_lu, D_lo, dnl.Dw])
    ], format='csc')

    sax_O = vstack([dnu.Ow, O_o, dnl.Ow], format='csc')
    sax_Cm = rs(list(sax_O.shape), w0=1).astype(bool)

    # Build main network
    dnm = DeepNetwork.from_matrices(
        'test_combine_main', sax_D, sax_I, sax_O, pn['c'], w0=pn['w0'], l0=pn['l0'], tau=pn['tau'], Cm=sax_Cm
    )

    return dnm


def build_main_network_2(pn, nm, no, dnm, I_o, D_om, O_o):
    # Build Main matrices:
    sax_I = hstack([dnm.Iw, I_o], format='csc')
    sax_D = vstack([hstack([dnm.Dw, csc((nm, no))]), hstack([D_om, csc((no, no))])], format='csc')
    sax_O = vstack([dnm.Ow, O_o], format='csc')
    sax_Cm = rs(list(sax_O.shape), w0=1).astype(bool)

    # Build main network
    dnm = DeepNetwork.from_matrices(
        'test_combine_main', sax_D, sax_I, sax_O, pn['c'], w0=pn['w0'], l0=pn['l0'], tau=pn['tau'], Cm=sax_Cm
    )

    return dnm


def build_sub_network(pn, no, ne, nso, nse, dnso, dnse, d_mapping):

    # build sub matrices
    sax_I = hstack([dnso.Iw, dnse.Iw], format='csc')
    sax_D = vstack([hstack([dnso.Dw, csc((nso, nse))]), hstack([csc((nse, nso)), dnse.Dw])], format='csc')
    sax_O = csc((nso + nse, no + ne))

    for i in range(2):
        d_map = {nid: l_d[i]['id'] for nid, l_d in sorted(d_mapping.items(), key=lambda x: x[0]) if len(l_d) > i}
        for j, (nid, oid) in enumerate(sorted(d_map.items(), key=lambda x: x[0])):
            sax_O[:, oid] = vstack([
                csc((nso * i, 1)),
                {0: dnso.O[:, j]}.get(i, dnse.O[:, min(j, dnse.O.shape[1] - 1)]),
                csc(((1 - i) * nse, 1))
            ], format='csc')

    # Build sub network and update mapping
    dns = DeepNetwork.from_matrices(
        'test_combine_sub', sax_D, sax_I, sax_O, pn['c'], w0=pn['w0'], l0=pn['l0'], tau=pn['tau']
    )
    d_mapping.update({'ido': range(nso), 'ide': range(nso, nso + nse)})

    return dns, d_mapping


def build_merged_network(pn, nu, nl, no, ne, nso, nse, dnu, dnl, D_lu, dnso, dnse, I_o, D_ou, D_lo, O_o):

    # Build merged matrices
    sax_I = hstack([dnso.Iw, I_o, dnu.Iw, dnl.Iw, I_o[:, :ne], dnse.Iw], format='csc')
    sax_D = vstack([
        hstack([dnso.Dw, dnso.Ow, csc((nso, nu)), csc((nso, nl)), csc((nso, ne)), csc((nso, nse))]),
        hstack([csc((no, nso)), csc((no, no)), D_ou, csc((no, nl)), csc((no, ne)), csc((no, nse))]),
        hstack([csc((nu, nso)), csc((nu, no)), dnu.Dw, csc((nu, nl)),  csc((nu, ne)), csc((nu, nse))]),
        hstack([csc((nl, nso)), D_lo, D_lu, dnl.Dw, D_lo[:, :ne], csc((nl, nse))]),
        hstack([csc((ne, nso)), csc((ne, no)), D_ou[:ne, :], csc((ne, nl)), csc((ne, ne)), csc((ne, nse))]),
        hstack([csc((nse, nso)), csc((nse, no)), csc((nse, nu)), csc((nse, nl)), dnse.Ow, dnse.Dw])
    ], format='csc')

    sax_O = vstack([csc((nso, pn['no'])), O_o, dnu.Ow, dnl.Ow, O_o[:ne, :], csc((nse, pn['no']))], format='csc')

    # Build sub network
    dnm = DeepNetwork.from_matrices(
        'test_combine_merged', sax_D, sax_I, sax_O, pn['c'], w0=pn['w0'], l0=pn['l0'], tau=pn['tau']
    )

    return dnm


def build_merged_network_2(pn, nm, no, ne, dnm, dns, I_o, D_om, O_o, d_mapping):
    l_oo, l_oe = [x[0]['id'] for _, x in sorted(d_mapping.items(), key=lambda t: t[0])], \
        [x[1]['id'] for _, x in sorted(d_mapping.items(), key=lambda t: t[0]) if len(x) > 1]

    # Build merged matrices
    sax_I = hstack([I_o + dns.IOw[:, l_oo], dnm.Iw, I_o[:, :ne] + dns.IOw[:, l_oe]], format='csc')
    sax_D = vstack([
        hstack([csc((no, no)), D_om, csc((no, ne))]),
        hstack([csc((nm, no)), dnm.Dw, csc((nm, ne))]),
        hstack([csc((ne, no)), D_om[:ne, :], csc((ne, ne))]),
    ], format='csc')

    sax_O = vstack([O_o, dnm.Ow, O_o[:ne, :]], format='csc')

    # Build sub network
    dnm = DeepNetwork.from_matrices(
        'test_combine_merged', sax_D, sax_I, sax_O, pn['c'], w0=pn['w0'], l0=pn['l0'], tau=pn['tau']
    )

    return dnm


def rs(l_dim, w0=5):
    return csc((np.random.randn(*l_dim) - 1.5 > 0) * w0)


def rl(n, l0):
    return DeepNetUtils.transform_level({i: np.random.choice([True, False]) for i in range(n)}, l0)

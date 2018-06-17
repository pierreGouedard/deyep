# Global imports
import unittest
import numpy as np
from scipy.sparse import csc_matrix

# Local import
from deyep.core.tools.equations import fnt, fot, fnp, fcp, bop, bnt, bit, bnp, bcu, bdu, bou, biu
from tests.comon import get_mat_from_path
from deyep.core.constructors.constructors import Constructor

__maintainer__ = 'Pierre Gouedard'


class TestEquations(unittest.TestCase):
    def setUp(self):

        np.random.seed(392)

        # Create a simple deep network (2 input nodes, 3 network nodes,, 2 output nodes)
        self.n_i, self.n_rn, self.n_o, self.capacity = 2, 5, 3, 2
        l_edges = [('input_0', 'network_0'), ('network_0', 'network_1'), ('network_0', 'network_2'),
                   ('network_1', 'output_0'), ('network_2', 'output_1')] +  \
                  [('input_0', 'network_3'), ('network_3', 'network_2')] + \
                  [('input_1', 'network_4'), ('network_4', 'output_2')] + \
                  [('input_1', 'network_2')]

        # Get matrices from list of edges and build network
        self.mat_in, self.mat_net, self.mat_out = get_mat_from_path(l_edges, self.n_i, self.n_rn, self.n_o)
        self.dn = Constructor.from_weighted_direct_matrices(self.mat_net, self.mat_in, self.mat_out, self.capacity)
        self.N = self.dn.network_nodes[0].frequency_stack.N
        # Create another very large to test complecity

    def test_forward_transmiting(self):
        """
        python -m unittest tests.core.equations.TestEquations.test_forward_transmiting

        """

        # Create forward messages
        l_input_active = [True, False]
        l_network_active = [True, False, False, True, False]
        sax_si, sax_sn = create_signals_forward(l_input_active, l_network_active, self.dn)

        # Test FNT
        res_fnt = fnt(self.dn.D, self.dn.I, sax_sn, sax_si)

        # Compute variable for test
        stack = self.dn.network_nodes[0].frequency_stack
        set_coefs = set(stack.coef_from_series(res_fnt.toarray()[:, 2], stack.basis_generic, n_jobs=0))
        coef_a = stack.coef_from_series(sax_sn.toarray()[:, 0], stack.basis_generic, n_jobs=0)[0]
        coef_b = stack.coef_from_series(sax_sn.toarray()[:, 3], stack.basis_generic, n_jobs=0)[0]

        # Assert expected result
        self.assertEqual(res_fnt.shape, (10, 5))
        self.assertTrue((res_fnt.toarray()[:, 0] == sax_si.toarray()[:, 0]).all())
        self.assertEqual(set_coefs, {coef_a, coef_b})

        # Create forward messages
        l_network_active = [False, True, True, False, False]
        _, sax_sn = create_signals_forward(l_input_active, l_network_active, self.dn)

        # Test FOT
        res_fot = fot(self.dn.O, sax_sn)

        # Assert expected result
        self.assertTrue((res_fot.toarray()[:, -1] == np.zeros(stack.N)).all())
        self.assertTrue((res_fot.toarray()[:, 0] == sax_sn.toarray()[:, 1]).all() &
                        (res_fot.toarray()[:, 1] == sax_sn.toarray()[:, 1]).all())

    def test_forward_processing(self):
        """
        python -m unittest tests.core.equations.TestEquations.test_forward_processing

        """

        l_input_active = [True, False]
        l_network_active = [True, False, False, True, False]
        sax_si, sax_sn = create_signals_forward(l_input_active, l_network_active, self.dn)

        # Get FNT
        res_fnt = fnt(self.dn.D, self.dn.I, sax_sn, sax_si)

        # Test FNP
        _ = fnp(res_fnt, self.dn.network_nodes)
        sax_C = fcp([n.active for n in self.dn.network_nodes], self.dn.Cm)

        self.assertEqual([n.active for n in self.dn.network_nodes], [True] * 4 + [False])
        self.assertEqual(len(self.dn.network_nodes[2].frequency_stack.map.items()[0][1]), 2)
        self.assertEqual(len(set(zip(sax_C.nonzero()[0], sax_C.nonzero()[1]))
                             .intersection(zip(self.dn.O.nonzero()[0], self.dn.O.nonzero()[1]))), 0)

    def test_backward_transmiting(self):
        """
        python -m unittest tests.core.equations.TestEquations.test_backward_transmiting

        """
        # Compute forward pass
        l_network_active = [False, True, True, False, False]
        sax_so, sax_C = compute_forward_pass(l_network_active, self.dn)
        l_output_got = [1, 0, 0]

        # Compute bop
        sax_sob = bop(sax_so, csc_matrix(l_output_got))

        sax_actives, sax_snb = csc_matrix(np.array([l_network_active]).repeat(self.dn.O.shape[1], axis=0)), \
            csc_matrix(np.zeros([self.N, len(l_network_active)]), dtype=np.complex)

        # Test BNT sax_bnp is empty
        sax_snb = bnt(self.dn.D, self.dn.O, sax_snb, sax_sob, sax_actives)

        for i, b in enumerate(l_network_active):
            if not b:
                self.assertTrue((sax_snb.toarray()[:, i] == np.zeros(self.N)).all())
            else:
                stack = self.dn.network_nodes[i].frequency_stack
                l_coefs_in, l_infos = stack.coef_from_series(sax_snb.toarray()[:, i], stack.basis_specific,
                                                             return_coef=True, n_jobs=0)

                self.assertTrue(len(l_coefs_in) > 0)
                self.assertTrue(all([c in [1, -1] for c in l_infos]))

        # Test BNT sax_bnp is not empty
        sax_snb = csc_matrix(np.zeros([self.N, len(l_network_active)], dtype=np.complex))
        sax_snb[:, 1] = sax_sob[:, 0]
        sax_snb[:, 2] = sax_sob[:, 1]

        sax_snb = bnt(self.dn.D, self.dn.O, sax_snb, sax_sob, sax_actives)

        self.assertEqual(len(sax_snb[0, :].nonzero()[1]), 3)

        stack = self.dn.network_nodes[3].frequency_stack
        _, l_infos = stack.coef_from_series(sax_snb.toarray()[:, 3], stack.basis_generic,
                                            return_coef=True, n_jobs=0)
        self.assertEqual(l_infos[0], -1)

        # Test BIT
        sax_snb[:, 2] = csc_matrix(np.zeros([self.N, 1], dtype=np.complex))

        sax_sib = bit(self.dn.I, sax_snb)

        stack = self.dn.input_nodes[0].frequency_stack
        _, l_infos = stack.coef_from_series(sax_sib.toarray()[:, 0], stack.basis_generic,
                                            return_coef=True, n_jobs=0)
        self.assertEqual(l_infos[0], -1)

    def test_backward_processing(self):
        """
        python -m unittest tests.core.equations.TestEquations.test_backward_processing

        """
        # Compute forward pass and backward transmiting
        l_network_active = [False, True, True, False, False]
        sax_so, sax_C = compute_forward_pass(l_network_active, self.dn)

        # Save candidate and active network nodes
        sax_Cb = sax_C.copy()
        sax_activation = csc_matrix(np.array([l_network_active]).repeat(self.dn.O.shape[1], axis=0))

        # Test BOP
        sax_sob = bop(sax_so, csc_matrix([1, 0, 0]), self.N)

        # Assert expected result
        feedback = np.zeros(sax_sob.shape[1])
        for i in np.unique(sax_sob.nonzero()[1]):
            feedback[i] = np.sign(sax_sob[0, i])

        self.assertTrue((feedback == np.array([1, -1, 0])).all())

        # add signal to test other processing
        sax_snb = csc_matrix(np.zeros([self.N, len(l_network_active)], dtype=np.complex))
        sax_snb[:, 1] = sax_sob[:, 0]
        sax_snb[:, 2] = sax_sob[:, 1]

        sax_snb = bnt(self.dn.D, self.dn.O, sax_snb, sax_sob, sax_activation)

        # Test BNP
        sax_snb = bnp(self.dn.network_nodes, sax_snb)

        # Assertion on expected result
        self.assertTrue((sax_snb.toarray()[:, 0] == np.zeros(self.N)).all())
        self.assertTrue((sax_snb.toarray()[:, 1] != np.zeros(self.N)).all() and sax_snb.toarray()[0, 1] > 0)
        self.assertTrue((sax_snb.toarray()[:, 2] != np.zeros(self.N)).all() and sax_snb.toarray()[0, 2] < 0)

    def test_network_update(self):
        """
         python -m unittest tests.core.equations.TestEquations.test_network_update

         """
        # Compute forward pass and backward transmiting
        l_network_active = [False, True, True, False, False]
        sax_so, sax_C = compute_forward_pass(l_network_active, self.dn)

        # Save candidate and active network nodes
        sax_Cb = sax_C.copy()
        sax_activation = csc_matrix(np.array([l_network_active]).repeat(self.dn.O.shape[1], axis=0))
        sax_got = csc_matrix([1, 0, 0])
        # Set node level
        for node in self.dn.network_nodes:
            node.level = 1

        # Create backward signals for output and network
        sax_sob = bop(sax_so, sax_got, self.N)

        sax_snb = csc_matrix(np.zeros([self.N, len(l_network_active)], dtype=np.complex))

        sax_snb[:, 1] = np.array([self.dn.network_nodes[0].frequency_stack.encode(sax_sob[:, 0].toarray()[:, 0])]).transpose()
        sax_snb[:, 2] = np.array([-1 * self.dn.network_nodes[0].frequency_stack.encode(sax_sob[:, 0].toarray()[:, 0])]).transpose()
        Dw_ = self.dn.Dw.copy()

        # Test BDU & BLU
        bdu(sax_snb, self.dn)

        self.assertEqual([node.level for node in self.dn.network_nodes], [1]*len(self.dn.network_nodes))
        self.assertTrue(self.dn.Dw[0, 1] == Dw_[0, 1] + 1)
        self.assertTrue(self.dn.Dw[0, 2] == Dw_[0, 2] - 1)

        bdu(sax_snb, self.dn, penalty=100)
        self.assertTrue(self.dn.D[0, 2] == 0)
        self.assertTrue(self.dn.network_nodes[2].level == 0)
        self.dn.deep_network['Dw'] = Dw_

        # Test BOU
        Ow_ = self.dn.Ow
        bou(sax_sob, sax_activation, self.dn)

        self.assertTrue(self.dn.Ow[1, 0] == Ow_[1, 0] + 1)
        self.assertTrue(self.dn.Ow[2, 1] == Ow_[2, 1] - 1)
        self.dn.deep_network['Ow'] = Ow_

        # Test BIU
        Iw_ = self.dn.Iw
        sax_snb[:, 0] = np.array([self.dn.input_nodes[0].frequency_stack.encode(sax_sob[:, 0].toarray()[:, 0])]).transpose()
        sax_snb[:, 2] = np.array([-1 * self.dn.input_nodes[1].frequency_stack.encode(sax_sob[:, 0].toarray()[:, 0])]).transpose()

        biu(sax_snb, self.dn)

        self.assertTrue(self.dn.Iw[0, 0] == Iw_[0, 0] + 1)
        self.assertTrue(self.dn.Iw[1, 2] == Iw_[1, 2] - 1)

        # Test BCU
        Cm_ = self.dn.Cm.copy()
        bcu(sax_got, sax_Cb, self.dn, w0=10)

        self.assertTrue(self.dn.Ow[sax_Cb] == 10)
        self.assertTrue((self.dn.Ow[~sax_Cb.toarray()] == Cm_[~sax_Cb.toarray()]).all())


def create_signals_forward(l_ia, l_na, dn):

    sax_si = csc_matrix([n.frequency_stack.fourrier_basis()[0] if l_ia[i] else np.zeros(n.frequency_stack.N)
                         for i, n in enumerate(dn.input_nodes)]).transpose()

    sax_sn = csc_matrix([n.frequency_stack.fourrier_basis()[0] if l_na[i] else np.zeros(n.frequency_stack.N)
                         for i, n in enumerate(dn.network_nodes)]).transpose()

    return sax_si, sax_sn


def compute_forward_pass(l_na, dn):

    # Generate Signal
    s = dn.input_nodes[0].frequency_stack.fourrier_basis()[0]
    sax_sn = csc_matrix([n.frequency_stack.encode(s) if l_na[i] else np.zeros(n.frequency_stack.N)
                         for i, n in enumerate(dn.network_nodes)]).transpose()

    # Compute forward transitting
    sax_so = fot(dn.O, sax_sn)
    sax_C = fcp(l_na, dn.O)

    return sax_so, sax_C



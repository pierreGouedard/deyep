# Global imports
import unittest
import numpy as np
from scipy.sparse import csc_matrix

# Local import
from deyep.core.tools.equations import fnt, fot, fnp, fcp, bop
from tests.comon import get_mat_from_path
from deyep.core.constructors.constructors import Constructor
from deyep.core.tools.linear_algebra import inner_product

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

        # Create another very large to test complecity

    def test_forward_transmission(self):
        """
        python -m unittest tests.core.equations.TestEquations.test_forward_transmission

        """

        # Create forward messages
        l_input_active = [True, False]
        l_network_active = [True, False, False, True, False]
        sax_si, sax_sn = create_signals_forward(l_input_active, l_network_active, self.dn)

        # Test FNT
        res_fnt = fnt(self.dn.D, self.dn.I, sax_sn, sax_si)

        # Compute variable for test
        stack = self.dn.network_nodes[0].frequency_stack
        set_coefs =  set(stack.coef_from_series(res_fnt.toarray()[:, 2], stack.basis_generic, n_jobs=0))
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
        sax_sn = fnp(res_fnt, self.dn.network_nodes)
        sax_C = fcp([n.active for n in self.dn.network_nodes], self.dn.O)

        self.assertEqual([n.active for n in self.dn.network_nodes], [True] * 4 + [False])
        self.assertEqual(len(self.dn.network_nodes[2].frequency_stack.map.items()[0][1]), 2)
        self.assertEqual(len(set(zip(sax_C.nonzero()[0], sax_C.nonzero()[1]))
                             .intersection(zip(self.dn.O.nonzero()[0], self.dn.O.nonzero()[1]))), 0)


    def test_backward_processing(self):
        """
        python -m unittest tests.core.equations.TestEquations.test_backward_processing

        """
        # Compute forward pass
        l_network_active = [False, True, True, False, False]
        sax_so, sax_C = compute_forward_pass(l_network_active, self.dn)
        l_output_got = [1, 0, 0]

        # Test BOP
        sax_bop = bop(sax_so, csc_matrix(l_output_got))

        # Test BCP

        # Test BNP

        # Test BLP

        raise NotImplementedError

    def test_backward_transmission(self):
        """
        python -m unittest tests.core.equations.TestEquations.test_backward_transmission

        """
        # Compute forward pass
        l_network_active = [False, True, True, False, False]
        sax_so, sax_C = compute_forward_pass(l_network_active, self.dn)
        l_output_got = [1, 0, 0]

        # Compute bop
        sax_bop = bop(sax_so, csc_matrix(l_output_got))

        # Assert expected result
        feedback = np.zeros(sax_bop.shape[1])
        for i in np.unique(sax_bop.nonzero()[1]):
            feedback[i] = np.sign(sax_bop[0, i])

        self.assertTrue((feedback == np.array([1, -1, 0])).all())

        import IPython
        IPython.embed()
        # Test BNT

        # Test BIT

        raise NotImplementedError



    def test_network_update(self):
        """
         python -m unittest tests.core.equations.TestEquations.test_network_update

         """

        # Test BDU

        # Test BOU

        # Test BIU

        raise NotImplementedError


def create_signals_forward(l_ia, l_na, dn):

    sax_si = csc_matrix([n.frequency_stack.fourrier_basis()[0] if l_ia[i] else np.zeros(n.frequency_stack.N)
                         for i, n in enumerate(dn.input_nodes)]).transpose()

    sax_sn = csc_matrix([n.frequency_stack.fourrier_basis()[0] if l_na[i] else np.zeros(n.frequency_stack.N)
                         for i, n in enumerate(dn.network_nodes)]).transpose()

    return sax_si, sax_sn


def create_signals_backward(l_ia, l_na, dn):

    sax_sob = csc_matrix([n.frequency_stack.fourrier_basis()[0] if l_ia[i] else np.zeros(n.frequency_stack.N)
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



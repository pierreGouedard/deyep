# Global imports
import unittest
import numpy as np
from scipy.sparse import csc_matrix

# Local import
from deyep.core.merger.comon import DeepNetMerger
from deyep.core.datastructures.deep_network_dry import DeepNetworkDry
from deyep.core.builder.comon import nodes_from_mat_dry
__maintainer__ = 'Pierre Gouedard'


class TestMerger(unittest.TestCase):
    # TODO: Make new test when merge strategy clear (obsolete test)
    def setUp(self):
        self.p = {'ni': 5, 'nd': 7, 'no': 2}

        self.sax_I_o = csc_matrix(
            [[True, True, False, False, False, False, False],
             [False, False, True, False, False, False, False],
             [True, True, False, True, False, False, False],
             [True, True, False, False, False, False, False],
             [False, False, True, True, False, False, False],
             ])

        self.sax_D_o = csc_matrix(
            [[False, False, False, False, True, False, True],
             [False, False, False, False, False, True, False],
             [False, False, False, False, True, True, True],
             [False, False, False, False, False, False, True],
             [False, False, False, False, False, False, False],
             [False, False, False, False, False, False, False],
             [False, False, False, False, False, False, False]
             ])

        self.sax_O_o = csc_matrix(
            [[False, False], [False, False], [False, False], [False, False],
             [True, False], [False, True], [False, True]])

        self.levels_o = [{1: True, 2: True, 3: True}, {1: True, 2: True, 3: True}, {1: True, 2: True, 3: True},
                         {1: False, 2: False, 3: False}, {1: True, 2: True, 3: True}, {1: True, 2: True, 3: True},
                         {1: True, 2: True, 3: True}]

        self.sax_I_c1 = csc_matrix(
            [[True, True, False, False, False, False, False],
             [False, False, True, False, False, False, False],
             [True, True, False, True, False, False, False],
             [True, True, False, False, False, False, False],
             [False, False, True, True, False, False, False],
             ])

        self.sax_D_c1 = csc_matrix(
            [[False, False, False, False, True, True, True],
             [False, False, False, False, False, False, False],
             [False, False, False, False, True, True, True],
             [False, False, False, False, False, False, False],
             [False, False, False, False, False, False, False],
             [False, False, False, False, False, False, False],
             [False, False, False, False, False, False, False]
             ])

        self.sax_O_c1 = csc_matrix(
            [[False, False], [False, False], [False, False], [False, False],
             [True, True], [False, False], [False, False]])

        self.levels_c1 = [{1: True, 2: True, 3: True}, {1: True, 2: True, 3: True}, {1: True, 2: True, 3: True},
                          {1: False, 2: False, 3: False}, {1: True, 2: True, 3: True}, {1: True, 2: True, 3: True},
                          {1: True, 2: True, 3: True}]

        self.sax_I_c2 = csc_matrix(
            [[True, False, False], [False, True, False], [True, False, False],
             [True, False, False], [False, True, False]])

        self.sax_D_c2 = csc_matrix(
            [[False, False, True], [False, False, True], [False, False, False]])

        self.sax_O_c2 = csc_matrix(
            [[False, False], [False, False], [True, True]])

        self.levels_c2 = [{1: True, 2: True, 3: True}, {1: True, 2: True, 3: True}, {1: True, 2: True, 3: True}]

        self.deep_network = DeepNetworkDry.from_matrices('test_merger',  self.sax_D_o, self.sax_I_o, self.sax_O_o,
                                                         levels=self.levels_o)

    def merge_network_step_1(self):
        """
        python -m unittest tests.core.merger.canonical.TestMerger.merge_network_step_1

        """

        dnm = DeepNetMerger([self.deep_network])

        sax_I, sax_D, sax_O,  = dnm.deep_network.I, dnm.deep_network.D, dnm.deep_network.O
        d_layers = dnm.classify_nodes_by_layer(sax_I, sax_D, dnm.deep_network.network_nodes)

        for _, l_nodes in sorted(d_layers.items(), key=lambda x: x[0]):
            sax_D, sax_O, _ = dnm.matrix_cleaning(l_nodes, sax_I, sax_D, sax_O)

        self.assertTrue((sax_D == self.sax_D_c1).toarray().all())
        self.assertTrue((sax_O == self.sax_O_c1).toarray().all())

        ax_active_nodes = (sax_O.sum(axis=1) > 0) | (sax_D.sum(axis=1) > 0)
        ax_active_nodes &= (sax_I.sum(axis=0) > 0).transpose() | (sax_D.sum(axis=0) > 0).transpose()

        self.assertTrue((ax_active_nodes == [[True], [False], [True], [False], [True], [False], [False]]).all())

    def merge_network_step_2(self):
        """
        python -m unittest tests.core.merger.canonical.TestMerger.merge_network_step_2

        """

        dnm = DeepNetMerger([self.deep_network]) \
            .clean_network()\
            .deep_network

        self.assertTrue(not (dnm.I != self.sax_I_c2).toarray().any())
        self.assertTrue(not (dnm.D != self.sax_D_c2).toarray().any())
        self.assertTrue(not (dnm.O != self.sax_O_c2).toarray().any())
        self.assertEqual([n.d_levels for n in dnm.network_nodes], self.levels_c2)

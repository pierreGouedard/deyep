# Global imports
import unittest
import numpy as np
from scipy.sparse import csc_matrix

from deyep.core.firing_graph.utils import mat_from_tuples
from deyep.core.firing_graph.graph import FiringGraph
from deyep.core.imputer.array import DoubleArrayImputer
from deyep.core.solver.sampler import Sampler

from deyep.utils.driver.nmp import NumpyDriver

__maintainer__ = 'Pierre Gouedard'


class TestSampler(unittest.TestCase):
    def setUp(self):

        self.ni, self.no = 100, 2
        self.N = 100
        self.selected_bits = {0: {0, 1, 2, 3}, 1: {50, 51, 52}}

        self.preselected_bits, bits = {0: {0, 1, 2, 3}, 1: {50, 51, 52}}, set(range(self.ni))
        for i in range(self.no):
            self.preselected_bits[i] = self.preselected_bits[i].union(set(np.random.choice(list(bits), 30)))
            bits = bits.difference(self.preselected_bits[i])

        self.input = csc_matrix(np.random.binomial(1, 0.1, (100, self.ni)), dtype=int)
        self.output = csc_matrix(np.random.binomial(1, 0.1, (100, self.no)), dtype=int)

    def sampler_init(self):
        """
        sampler test for first usage (initialisation)

        python -m unittest tests.core.sampler.TestSampler.sampler_init

        """
        # Create simple imputer and sampler
        imputer = init_imputer(self.input, self.output)
        sampler = Sampler((self.ni, self.no), self.N, imputer)

        # sample bits
        sampler.sample_supervised()

        # Make sure sampling is correct
        for i in sampler.preselect_bits[0]:
            self.assertTrue(self.input[:, i].transpose().dot(self.output[:, 0]) > 0)
        for i in sampler.preselect_bits[1]:
            self.assertTrue(self.input[:, i].transpose().dot(self.output[:, 1]) > 0)

        # build firing graph for drainer
        sampler.build_graph_multiple_output()

        # Test dim of Firing graph
        self.assertEqual(len(sampler.firing_graph.core_vertices), 2)
        self.assertEqual(len(sampler.firing_graph.input_vertices), self.ni)
        self.assertEqual(len(sampler.firing_graph.output_vertices), self.no)

        # Test level of core vertices
        self.assertTrue(all([v.l0 == 1 for v in sampler.firing_graph.core_vertices]))

    def sampler_main(self):
        """
        sampler test for main usage of sampler (after initialisation)

        python -m unittest tests.core.sampler.TestSampler.sampler_main

        """
        # Create simple imputer and sampler
        imputer = init_imputer(self.input, self.output)
        sampler = Sampler((self.ni, self.no), self.N, imputer, self.selected_bits, self.preselected_bits)

        # sample bits
        sampler.sample_supervised()

        # Make sure sampling is correct
        for i in sampler.preselect_bits[0]:
            self.assertTrue(self.input[:, i].transpose().dot(self.output[:, 0]) > 0)
        for i in sampler.preselect_bits[1]:
            self.assertTrue(self.input[:, i].transpose().dot(self.output[:, 1]) > 0)

        # build firing graph for drainer
        sampler.build_graph_multiple_output()

        # Test dim of Firing graph
        self.assertEqual(len(sampler.firing_graph.core_vertices), 6)
        self.assertEqual(len(sampler.firing_graph.input_vertices), self.ni)
        self.assertEqual(len(sampler.firing_graph.output_vertices), self.no)

        # Test level of core vertices
        for i in range(self.no):
            l_v = sampler.firing_graph.core_vertices[i * 3: (i+1) * 3]
            self.assertEqual([v.l0 for v in l_v], [1, len(self.selected_bits[i]), 2])


def init_imputer(ax_input, ax_output):
    # Create temporary directory for test
    driver = NumpyDriver()
    tmpdirin, tmpdirout = driver.TempDir('test_sampler', suffix='in', create=True), \
                          driver.TempDir('test_sampler', suffix='out', create=True)

    # Create I/O and save it into tmpdir files
    driver.write_file(ax_input, driver.join(tmpdirin.path, 'forward.npz'), is_sparse=True)
    driver.write_file(ax_output, driver.join(tmpdirin.path, 'backward.npz'), is_sparse=True)

    # Create and init imputer
    imputer = DoubleArrayImputer('test', tmpdirin.path, tmpdirout.path)
    imputer.read_raw_data('forward.npz', 'backward.npz')
    imputer.run_preprocessing()
    imputer.write_features('forward.npz', 'backward.npz')
    imputer.stream_features()

    tmpdirin.remove()
    tmpdirout.remove()

    return imputer
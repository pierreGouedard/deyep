# Global imports
import unittest
import numpy as np
from scipy.sparse import csc_matrix

# Local import
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
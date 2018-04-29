# Global imports
import numpy as np
import unittest
from scipy.sparse import lil_matrix
import time

# Local import
from deyep.core.constructors.constructors import Constructor
from deyep.core.tools.linear_algebra import inner_product, get_fourrier_coef, get_fourrier_series

__maintainer__ = 'Pierre Gouedard'


class TestOperators(unittest.TestCase):
    def setUp(self):

        # Create a simple deep network (2 input nodes, 3 network nodes,, 2 output nodes)
        self.N = 100

        self.vect_fourrier = np.array([get_fourrier_coef(self.N, k) for k in range(self.N)])
        self.mat_fourrier = self.vect_fourrier.reshape((10, 10))

        self.vector_bool = np.random.randint(0, 2, 100).astype(bool)
        self.mat_bool = self.vector_bool.reshape(10, 10)

        self.vector_real = np.random.randint(0, 20, 100)
        self.mat_real = self.vector_real.reshape(10, 10)

    def test_operators_fourrier(self):
        """
        Test basic frequency management
        python -m unittest tests.core.operators.TestOperators.test_operators_fourrier

        """

        i, j = np.random.randint(0, 100), np.random.randint(0, 100)
        x, y = self.vect_fourrier[i], self.vect_fourrier[j]
        # Test Multiplication
        res = (1. / np.linalg.norm(x)) * x * y
        self.assertAlmostEqual(np.real(self.vect_fourrier[(i + j) % 100] - res), 0, delta=1e-10)
        self.assertAlmostEqual(np.imag(self.vect_fourrier[(i + j) % 100] - res), 0, delta=1e-10)

        # Test inner product
        res_x = np.real(inner_product(get_fourrier_series(x), get_fourrier_series(x)))
        res_y = np.real(inner_product(get_fourrier_series(y), get_fourrier_series(y)))
        res_ = np.real(inner_product(get_fourrier_series(y), get_fourrier_series(x)))
        _res = np.real(inner_product(get_fourrier_series(x), get_fourrier_series(y)))

        self.assertAlmostEqual(res_x, 1.0, delta=1e-9)
        self.assertAlmostEqual(res_y, 1.0, delta=1e-9)
        self.assertAlmostEqual(res_, 0.0, delta=1e-9)
        self.assertAlmostEqual(_res, 0.0, delta=1e-9)
        self.assertAlmostEqual(res_, _res, delta=1e-9)

        # Test running time
        x_, y_ = get_fourrier_coef(500000, i), get_fourrier_coef(500000, j)

        t0 = time.time()
        _ = np.real(inner_product(get_fourrier_series(x_), get_fourrier_series(y_)))
        print('time for inner product in F_500000 is {} seconds'.format(time.time() - t0))

        import IPython
        IPython.embed()
        # Test matrix multiplication in F_N


        # Test matrix multiplication in from F_N to R

        # Test Chi in F_N

        # Test Upsilon in F_N

    def test_operators_bool(self):
        """
        Test basic frequency management
        python -m unittest tests.core.operators.TestOperators.test_operators_basics

        """
        import IPython
        IPython.embed()
        # Test addition

        # Test Multiplication

        # Test inner product

        # Test matrix multiplication in B

    def test_operators_real(self):
        """
        Test basic frequency management
        python -m unittest tests.core.operators.TestOperators.test_operators_basics

        """
        import IPython
        IPython.embed()
        # Test addition

        # Test Multiplication

        # Test inner product

        # Test matrix multiplication in R

        # Test Chi in R

        # Test Upsilon in R



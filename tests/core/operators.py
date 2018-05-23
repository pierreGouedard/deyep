# Global imports
import numpy as np
import unittest
from scipy.sparse import csr_matrix, csc_matrix
import time

# Local import
from deyep.core.tools.linear_algebra import inner_product, get_fourrier_coef_from_params, get_fourrier_params, \
    get_fourrier_series, vector_product_type_1, vector_product_type_2, vector_product_type_3, matrix_product, matrix_fourrier_product, \
    vector_fourrier_diag, Chi_fourrier,  Upsilon_fourrier

__maintainer__ = 'Pierre Gouedard'


class TestOperators(unittest.TestCase):
    def setUp(self):

        np.random.seed(3203)
        # Create a simple deep network (2 input nodes, 3 network nodes,, 2 output nodes)
        self.N = 1002528
        self.n = int(np.sqrt(self.N / 2))
        self.sparsity = 0.1

        self.vect_fourrier = np.array([get_fourrier_coef_from_params(self.N, k) for k in range(self.N / 2)])
        self.vect_fourrier[np.random.choice(range(self.N / 2), int((self.N / 2) * self.sparsity), replace=False)] = 0.
        self.mat_fourrier = self.vect_fourrier.reshape((self.n, self.n))

    def test_basic_fourrier_operators(self):
        """
        Test basic frequency management
        python -m unittest tests.core.operators.TestOperators.test_basic_fourrier_operators
        """
        i, j = np.random.randint(0, self.N / 2), np.random.randint(0, self.N / 2)
        x, y = self.vect_fourrier[i], self.vect_fourrier[j]

        # Test Multiplication
        res = (1. / np.linalg.norm(x)) * x * y
        self.assertAlmostEqual(np.real(self.vect_fourrier[(i + j) % self.N] - res), 0, delta=1e-10)
        self.assertAlmostEqual(np.imag(self.vect_fourrier[(i + j) % self.N] - res), 0, delta=1e-10)

        # Test inner product
        t0 = time.time()
        res_x = np.real(inner_product(get_fourrier_series(x), get_fourrier_series(x)))
        res_y = np.real(inner_product(get_fourrier_series(y), get_fourrier_series(y)))
        res_ = np.real(inner_product(get_fourrier_series(y), get_fourrier_series(x)))
        _res = np.real(inner_product(get_fourrier_series(x), get_fourrier_series(y)))
        print('time for inner product in R is {} seconds'.format((time.time() - t0) / 4.))

        self.assertAlmostEqual(res_x, 1.0, delta=1e-9)
        self.assertAlmostEqual(res_y, 1.0, delta=1e-9)
        self.assertAlmostEqual(res_, 0.0, delta=1e-9)
        self.assertAlmostEqual(_res, 0.0, delta=1e-9)
        self.assertAlmostEqual(res_, _res, delta=1e-9)

        # Test Chi in F_N
        coefs = np.array([get_fourrier_coef_from_params(self.N, k + np.random.randint(0, self.N / 2 - k - 1))
                          for k in range(10)])
        freq_map, x1, x2 = dict(zip(coefs, [1] * 10)), 2 * coefs[4], -3 * coefs[6]

        t0 = time.time()
        res = Chi_fourrier(x1, coefs, freq_map=freq_map)
        self.assertEqual(res, 1)
        res = Chi_fourrier(x2, coefs, freq_map=freq_map)
        self.assertEqual(res, 0)
        print('SINGLEPROCESS get Chi in F_{} is {} seconds'.format(self.N, (time.time() - t0) / 2))

        # Test Upsilon in F_N
        t0 = time.time()
        res = Upsilon_fourrier(x1, coefs, freq_map=freq_map)
        self.assertEqual(res, 1)
        res = Upsilon_fourrier(x2, coefs, freq_map=freq_map)
        self.assertEqual(res, -1)
        print('SINGLEPROCESS get Upsilon in F_{} is {} seconds'.format(self.N, (time.time() - t0) / 2))

        t0 = time.time()
        res = Chi_fourrier(x1, coefs, freq_map=freq_map, n_jobs=0)
        self.assertEqual(res, 1)
        res = Chi_fourrier(x2, coefs, freq_map=freq_map, n_jobs=0)
        self.assertEqual(res, 0)
        print('MULTIPROCESS get Chi in F_{} is {} seconds'.format(self.N, (time.time() - t0) / 2))

        # Test Upsilon in F_N
        t0 = time.time()
        res = Upsilon_fourrier(x1, coefs, freq_map=freq_map, n_jobs=0)
        self.assertEqual(res, 1)
        res = Upsilon_fourrier(x2, coefs, freq_map=freq_map, n_jobs=0)
        self.assertEqual(res, -1)
        print('MULTIPROCESS get Upsilon in F_{} is {} seconds'.format(self.N, (time.time() - t0) / 2))

    def test_basic_matrix_fourrier_operators(self):
        """
        Test basic frequency management
        python -m unittest tests.core.operators.TestOperators.test_basic_matrix_fourrier_operators

        """

        # Type 1 operation; vector - vector = vector (1 core) SPARSE
        t0 = time.time()
        for i in range(10):
            res = vector_product_type_1(csc_matrix(self.vect_fourrier), csc_matrix(self.vect_fourrier))
        print('Mean time for vector product in F_{} is {} seconds'.format(self.N, (time.time() - t0) / 10.))

        # Type 2 vector - vector => matrix (1 cores) SPARSE
        vect_x, vect_y = csc_matrix(self.vect_fourrier[:self.n * 10]), csc_matrix(self.vect_fourrier[:(self.n * 10)])
        t0 = time.time()

        for i in range(10):
            res = vector_product_type_2(vect_x, vect_y)

        print('Mean time for vector type 2 product in F_{} is {} seconds'.format(self.N, (time.time() - t0) / 10.))

        self.assertEqual(res.shape,  (vect_x.shape[1], vect_y.shape[1]))
        self.assertEqual(get_fourrier_params(np.sqrt(self.N) * res[289, 472]), (self.N, 761))

        # Type 3 vector matrix => vector (1 core) SPARSE
        vect_x, mat_x = csc_matrix(self.vect_fourrier[:self.n]), csc_matrix(self.mat_fourrier)
        t0 = time.time()
        for i in range(10):
            res = vector_product_type_3(vect_x, mat_x)

        print('Mean time for vector - matrix product in F_{} is {} seconds'.format(self.N, (time.time() - t0) / 10.))
        self.assertEqual(res.shape, (1, mat_x.shape[-1]))

        # Type 4 matrix - matrix => matrix (all core) NO SPARSE
        t0 = time.time()
        mat_x, mat_y = csc_matrix(self.mat_fourrier), csc_matrix(self.mat_fourrier)
        for i in range(10):
            res = matrix_product(mat_x, mat_y)

        print('Mean time for matrix - matrix product in F_{} is {} seconds'.format(self.N, (time.time() - t0) / 10.))
        self.assertEqual(res.shape, mat_x.shape)

    def test_advanced_matrix_fourrier_operators(self):
        """
        Test basic frequency management
        python -m unittest tests.core.operators.TestOperators.test_advanced_matrix_fourrier_operators
        """

        # Test matrix_fourrier_product
        mat_x, mat_y = csc_matrix(self.mat_fourrier[:7, :7]), csc_matrix(self.mat_fourrier[:7, :7])
        t0 = time.time()
        res = matrix_fourrier_product(mat_x, mat_y)
        print('Mean time for matrix - matrix fourrier product in F_{} is {} seconds'
              .format(self.N, (time.time() - t0)))

        self.assertEqual(res.nnz, len(mat_x.diagonal().nonzero()[0]))

        # Test matrix_fourrier_product MULTIPROCESS
        t0 = time.time()
        res = matrix_fourrier_product(mat_x, mat_y, n_jobs=0)
        print('MULTIPROCESS Mean time for matrix - matrix fourrier product in F_{} is {} seconds'
              .format(self.N, (time.time() - t0)))

        self.assertEqual(res.nnz, len(mat_x.diagonal().nonzero()[0]))

        # Test vector_fourrier_type_2
        vect_x, vect_y = csc_matrix(self.vect_fourrier), csc_matrix(self.vect_fourrier)

        t0 = time.time()
        res = vector_fourrier_diag(vect_x[0, :1000], vect_y[0, :1000], n_jobs=0)
        print('MULTIPROCESS get series in F_{} is {} seconds'.format(self.N, (time.time() - t0)))

        self.assertTrue((res.nonzero()[1] == vect_x[0, :1000].nonzero()[1]).all())

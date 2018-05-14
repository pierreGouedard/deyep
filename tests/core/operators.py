# Global imports
import numpy as np
import unittest
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
import time

# Local import
from deyep.core.constructors.constructors import Constructor
from deyep.core.tools.linear_algebra import inner_product, get_fourrier_coef, get_fourrier_key, get_fourrier_series, \
    vector_product_type_1, vector_product_type_2, vector_product_type_3, matrix_product, matrix_fourrier_product, \
    vector_fourrier_diag, Chi_fourrier,  Upsilon_fourrier

__maintainer__ = 'Pierre Gouedard'


class TestOperators(unittest.TestCase):
    def setUp(self):

        np.random.seed(303)
        # Create a simple deep network (2 input nodes, 3 network nodes,, 2 output nodes)
        self.N = 1000000
        self.n = int(np.sqrt(self.N))
        self.sparsity = 0.8

        self.vect_fourrier = np.array([get_fourrier_coef(self.N, k) for k in range(self.N)])
        self.vect_fourrier[np.random.choice(range(self.N), int(self.N * self.sparsity), replace=False)] = 0.
        self.mat_fourrier = self.vect_fourrier.reshape((self.n, self.n))

        self.vector_bool = np.random.randint(0, 2, self.N).astype(bool)
        self.vector_bool[np.random.choice(range(self.N), int(self.N * self.sparsity), replace=False)] = 0.
        self.mat_bool = self.vector_bool.reshape(self.n, self.n)

        self.vector_real = np.random.randint(0, 20, self.N)
        self.vector_real[np.random.choice(range(self.N), int(self.N * self.sparsity), replace=False)] = 0.
        self.mat_real = self.vector_real.reshape(self.n, self.n)

    def test_operators_fourrier(self):
        """
        Test basic frequency management
        python -m unittest tests.core.operators.TestOperators.test_operators_fourrier

        """
        # i, j = np.random.randint(0, self.N), np.random.randint(0, self.N)
        # x, y = self.vect_fourrier[i], self.vect_fourrier[j]
        #
        # # Test Multiplication
        # res = (1. / np.linalg.norm(x)) * x * y
        # self.assertAlmostEqual(np.real(self.vect_fourrier[(i + j) % self.N] - res), 0, delta=1e-10)
        # self.assertAlmostEqual(np.imag(self.vect_fourrier[(i + j) % self.N] - res), 0, delta=1e-10)
        #
        # # Test inner product
        # t0 = time.time()
        # res_x = np.real(inner_product(get_fourrier_series(x), get_fourrier_series(x)))
        # res_y = np.real(inner_product(get_fourrier_series(y), get_fourrier_series(y)))
        # res_ = np.real(inner_product(get_fourrier_series(y), get_fourrier_series(x)))
        # _res = np.real(inner_product(get_fourrier_series(x), get_fourrier_series(y)))
        # print('time for inner product in R is {} seconds'.format((time.time() - t0) / 4.))
        #
        # self.assertAlmostEqual(res_x, 1.0, delta=1e-9)
        # self.assertAlmostEqual(res_y, 1.0, delta=1e-9)
        # self.assertAlmostEqual(res_, 0.0, delta=1e-9)
        # self.assertAlmostEqual(_res, 0.0, delta=1e-9)
        # self.assertAlmostEqual(res_, _res, delta=1e-9)

        # Type 1 operation; vector - vector = vector (1 core) SPARSE
        t0 = time.time()
        for i in range(10):
            res = vector_product_type_1(lil_matrix(self.vect_fourrier), lil_matrix(self.vect_fourrier))
        print('Mean time for vector product in F_{} is {} seconds'.format(self.N, (time.time() - t0) / 10.))

        # # Type 2 vector - vector => matrix (1 cores) SPARSE
        # vect_x, vect_y = lil_matrix(self.vect_fourrier[:self.n * 10]), lil_matrix(self.vect_fourrier[:(self.n * 10)])
        # t0 = time.time()
        #
        # for i in range(10):
        #     res = vector_product_type_2(vect_x, vect_y)
        #
        # print('Mean time for vector type 2 product in F_{} is {} seconds'.format(self.N, (time.time() - t0) / 10.))
        #
        # self.assertEqual(res.shape,  (vect_x.shape[1], vect_y.shape[1]))
        # self.assertEqual(get_fourrier_key(np.sqrt(self.N) * res[289, 472]), (self.N, 761))
        #
        # # Type 3 vector matrix => vector (1 core) SPARSE
        # vect_x, mat_x = lil_matrix(self.vect_fourrier[:self.n]), lil_matrix(self.mat_fourrier)
        # t0 = time.time()
        # for i in range(10):
        #     res = vector_product_type_3(vect_x, mat_x)
        #
        # print('Mean time for vector - matrix product in F_{} is {} seconds'.format(self.N, (time.time() - t0) / 10.))
        # self.assertEqual(res.shape, (1, mat_x.shape[-1]))
        #
        # # Type 4 matrix - matrix => matrix (all core) NO SPARSE
        # t0 = time.time()
        # mat_x, mat_y = lil_matrix(self.mat_fourrier), lil_matrix(self.mat_fourrier)
        # for i in range(10):
        #     res = matrix_product(mat_x, mat_y)
        #
        # print('Mean time for matrix - matrix product in F_{} is {} seconds'.format(self.N, (time.time() - t0) / 10.))
        # self.assertEqual(res.shape, mat_x.shape)
        #
        # # Test matrix_fourrier_product
        # mat_x, mat_y = lil_matrix(self.mat_fourrier[:7, :7]), lil_matrix(self.mat_fourrier[:7, :7])
        # t0 = time.time()
        # res = matrix_fourrier_product(mat_x, mat_y)
        # print('Mean time for matrix - matrix fourrier product in F_{} is {} seconds'
        #       .format(self.N, (time.time() - t0)))
        #
        # self.assertEqual(res.nnz, len(mat_x.diagonal().nonzero()[0]))
        #
        # # Test matrix_fourrier_product MULTIPROCESS
        # t0 = time.time()
        # res = matrix_fourrier_product(mat_x, mat_y, n_jobs=0)
        # print('MULTIPROCESS Mean time for matrix - matrix fourrier product in F_{} is {} seconds'
        #       .format(self.N, (time.time() - t0)))
        #
        # self.assertEqual(res.nnz, len(mat_x.diagonal().nonzero()[0]))
        #
        # # Test vector_fourrier_type_2
        # vect_x, vect_y = lil_matrix(self.vect_fourrier), lil_matrix(self.vect_fourrier)
        #
        # t0 = time.time()
        # res = vector_fourrier_diag(vect_x[0, :1000], vect_y[0, :1000], n_jobs=0)
        # print('MULTIPROCESS get series in F_{} is {} seconds'.format(self.N, (time.time() - t0)))
        #
        # self.assertTrue((res.nonzero()[1] == vect_x[0, :1000].nonzero()[1]).all())
        import IPython
        IPython.embed()

        # Test Chi in F_N
        coefs = np.array([get_fourrier_coef(self.N, k + np.random.randint(0, self.N - k - 1)) for k in range(10)])
        freq_map, x = dict(zip(coefs, [1] * 10)), np.ones(10)
        x[np.random.choice(range(10), 5, replace=False)] = -1

        t0 = time.time()
        res = Chi_fourrier(coefs.dot(x), coefs, freq_map=freq_map)
        print('SINGLEPROCESS get Chi in F_{} is {} seconds'.format(self.N, (time.time() - t0)))

        self.assertEqual(res, 0.)

        t0 = time.time()
        res = Chi_fourrier(coefs.dot(x), coefs, n_jobs=0, freq_map=freq_map)
        print('SINGLEPROCESS get Chi in F_{} is {} seconds'.format(self.N, (time.time() - t0)))

        self.assertEqual(res, 0.)


        x, y = Chi_fourrier(coefs.dot(x), coefs, freq_map=d_coefs)


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

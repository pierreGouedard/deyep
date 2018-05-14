# Global import
import numpy as np
from scipy.sparse import lil_matrix
from pathos.multiprocessing import ProcessingPool as Pool, cpu_count


class InnerProductParallel(object):
    def __init__(self, keep_real=False, round=False):
        self.keep_real = keep_real
        self.round = round

    def f(self, t):
        res = inner_product(get_fourrier_series(t[0]), get_fourrier_series(t[1]))

        if self.keep_real:
            res = np.real(res)

        if self.round:
            res = np.round(res)

        return res


class ChiFourrierParallel(object):
    def __init__(self, freq_map=None):
        self.freq_map = freq_map

    def f(self, t):

        res = Chi(np.round(np.real(inner_product(get_fourrier_series(t[0]), get_fourrier_series(t[1])))))

        if self.freq_map is not None:
            res *= self.freq_map[t[1]]
        else:
            res *= t[1]

        return res


class UpsilonFourrierParallel(object):
    def __init__(self, freq_map=None):
        self.freq_map = freq_map

    def f(self, t):

        res = Upsilon(np.round(np.real(inner_product(get_fourrier_series(t[0]), get_fourrier_series(t[1])))))

        if self.freq_map is not None:
            res *= self.freq_map[t[1]]
        else:
            res *= t[1]

        return res


def get_fourrier_coef(N, k):
    return np.exp(np.complex(np.log(1. / np.sqrt(N)), - (2. * np.pi * k) / N))


def get_fourrier_key(coef):

    # Get N
    N = int(np.round(1. / pow(np.linalg.norm(coef), 2)))

    # Get k
    k = (np.angle(coef) * N) / (2. * np.pi)

    if k <= 0:
        k = int(np.round(- k))
    else:
        k = int(np.round(N - k))

    return N, k


def get_fourrier_series(coef):

    # Get N
    N, k = get_fourrier_key(coef)

    return coef * np.exp(-1j * (2. * np.pi * k * np.arange(-1, N - 1)) / N)


def inner_product(x, y):
    """
    x and y are 1d numpy array of the same size

    :param x:
    :param y:
    :return:
    """
    return x.dot(y.conjugate())


def vector_product_type_1(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    try:
        return x.toarray()[0].dot(y.toarray()[0])
    except MemoryError:
        return x.dot(y.tocsc().transpose())[0, 0]


def vector_product_type_2(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    return x.tocsc().transpose().dot(y)


def vector_product_type_3(x, A):
    """

    :param x:
    :param A:
    :return:
    """
    try:
        return lil_matrix(x.toarray()[0].dot(A.toarray()))
    except MemoryError:
        return x.dot(A)


def matrix_product(A, B):
    """

    :param A:
    :param B:
    :return:
    """
    n, d = float(A.nnz * np.prod(B.shape) + B.nnz * np.prod(A.shape)), np.prod(A.shape) * np.prod(B.shape)
    sparsity = 1 - (n / d)

    if sparsity >= 0.6:
        return A.dot(B)
    else:
        try:
            return lil_matrix(A.toarray().dot(B.toarray()))
        except MemoryError:
            return A.dot(B)


def matrix_fourrier_product(A, B, n_jobs=1):
    """
    To PARALLELIZE
    :param A:
    :param B:
    :param method:
    :return:
    """

    # Build Pool if necessary
    if n_jobs != 1:
        p = Pool({0: cpu_count()}.get(n_jobs, n_jobs))

    (l_xa, l_ya), (l_xb, l_yb), res = A.nonzero(), B.nonzero(), lil_matrix((A.shape[0], B.shape[1]))
    for i in set(l_xa):
        for j in set(l_yb):
            l_indices = list(set(l_ya[l_xa == i]).intersection(set(l_xb[l_yb == j])))

            # multiprocess version
            if n_jobs != 1:
                # Instantiate class that implement inner product
                innerp = InnerProductParallel(keep_real=True, round=True)

                # Parallel inner product
                res_ = sum(p.map(innerp.f, zip(A[i, l_indices].toarray()[0], B[l_indices, j].toarray()[:, 0])))

            # single process version
            else:

                res_ = 0
                for k in l_indices:
                    res_ += np.round(np.real(inner_product(get_fourrier_series(A[i, k]), get_fourrier_series(B[k, j]))))

            if res_ > 0:
                res[i, j] = res_

    return res


def vector_fourrier_diag(x, y, n_jobs=1):
    """
    To PARALLELIZE

    :param x:
    :param y:
    :param method:
    :return:
    """

    res, l_indices = lil_matrix(x.shape), list(set(x.nonzero()[1]).intersection(y.nonzero()[1]))

    if n_jobs != 1:

        p = Pool({0: cpu_count()}.get(n_jobs, n_jobs))

        # Instantiate class that implement inner product
        innerp = InnerProductParallel(keep_real=True, round=True)

        # Parallel inner product
        res_ = lil_matrix(p.map(innerp.f, zip(x[0, l_indices].toarray()[0], y[0, l_indices].toarray()[0])))

    else:
        # Loop on common non zeros diag indices
        res_ = []
        for i in l_indices:
            res_ += [inner_product(get_fourrier_series(x[0, i]), get_fourrier_series(y[0, i]))]

        res_ = lil_matrix(res_)

    # Set result sparse matrix
    res[0, l_indices] = res_

    return res


def Chi(x):
    """

    :param x:
    :return:
    """
    if x > 0:
        return 1
    else:
        return 0


def Chi_fourrier(x, l_coefs, n_jobs=1, freq_map=None):

    if n_jobs != 1:

        p = Pool({0: cpu_count()}.get(n_jobs, n_jobs))

        # Instantiate class that implement inner product
        chip = ChiFourrierParallel(freq_map=freq_map)

        # Parallel inner product
        res = sum(p.map(chip.f, zip([x] * len(l_coefs), l_coefs)))

    else:
        res = 0
        # set freq map to identity in case None
        if freq_map is None:
            freq_map = dict(zip(l_coefs, l_coefs))

        for c in l_coefs:
            res += Chi(np.round(np.real(inner_product(get_fourrier_series(x), get_fourrier_series(c))))) * \
                   freq_map[c]

    return res


def Upsilon(x):
    """

    :param x:
    :return:
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def Upsilon_fourrier(x, l_coefs, n_jobs=1, freq_map=None):

    if n_jobs != 1:

        p = Pool({0: cpu_count()}.get(n_jobs, n_jobs))

        # Instantiate class that implement inner product
        chip = UpsilonFourrierParallel(freq_map=freq_map)

        # Parallel inner product
        res = sum(p.map(chip.f, zip([x] * len(l_coefs), l_coefs)))

    else:
        res = 0
        # set freq map to identity in case None
        if freq_map is None:
            freq_map = dict(zip(l_coefs, l_coefs))

        for c in l_coefs:
            res += Upsilon(np.round(np.real(inner_product(get_fourrier_series(x), get_fourrier_series(c))))) * \
                   freq_map[c]

    return res
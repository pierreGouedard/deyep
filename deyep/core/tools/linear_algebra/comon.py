import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix


def Upsilon(x):
    """
    :param x:
    :return:
    """
    return np.sign(x)


def Chi(x):
    """

    :param x:
    :return:
    """
    if x > 0:
        return 1
    else:
        return 0


def Chi_ax(ax_v):

    # Vectorize discretie value function
    vdicretizer = np.vectorize(lambda x: Chi(x))

    # Apply to array
    ax_v_ = vdicretizer(ax_v)

    return ax_v_


def Upsilon_ax(ax_v):

    # Vectorize discretie value function
    vdicretizer = np.vectorize(lambda x: Upsilon(x))

    # Apply to array
    ax_v_ = vdicretizer(ax_v)

    return np.sign(ax_v)


def Chi_sax(sax_v):
    return sax_v > 0


def Upsilon_sax(sax_v):
    if isinstance(sax_v, csc_matrix):
        return csc_matrix((np.sign(sax_v.data), sax_v.indices, sax_v.indptr), shape=sax_v.shape)
    elif isinstance(sax_v, csr_matrix):
        return csr_matrix((np.sign(sax_v.data), sax_v.indices, sax_v.indptr), shape=sax_v.shape)


def get_key_from_series(sax_s, set_keys=None, return_coef=False):
    N, set_indices, l_res, l_coefs = sax_s.shape[-1], set(sax_s.nonzero()[1]), [], []

    if set_keys is not None:
        set_indices = set_keys.intersection(set_indices)

    for i in set_indices:
        l_res += [(i, N)]
        l_coefs += [sax_s[0, i]]

    if return_coef:
        return l_res, l_coefs
    else:
        return l_res


def get_canonical_basis(N, k):
    sax_m = lil_matrix((1, N), dtype=int)
    sax_m[0, k] = 1
    return sax_m


def get_canonical_signal(N, d_keys):
    sax_m = lil_matrix((1, N), dtype=int)
    for k, v in d_keys.items():
        sax_m[0, k] = v
    return sax_m


def inner_product(x, y):
    """
    x and y are 1d numpy array of the same size

    :param x:
    :param y:
    :return:
    """
    return x.dot(y.transpose())[0, 0]

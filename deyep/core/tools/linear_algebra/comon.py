import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


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


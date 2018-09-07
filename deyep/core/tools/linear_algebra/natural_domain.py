# Global import
from scipy.sparse import lil_matrix


def get_key_from_series(sax_s, set_keys=None, return_coef=False):
    N, set_indices, l_res, l_coefs = sax_s.shape[-1], set(sax_s.nonzero()[1]), [], []

    if set_keys is not None:
        set_indices = set_keys.intersection(set_indices)

    for i in set_indices:
        l_res += [(N, i)]
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

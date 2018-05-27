# Global import
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool, cpu_count
from scipy.sparse import csc_matrix
# Local import
from deyep.core.tools import linear_algebra as la
from deyep.utils.names import KVName


class FNT(object):
    def __init__(self, shape):
        self.shape = shape

    def f(self, t):
        res = []
        if len(t[1].data) > 0:
            res = list(t[0].dot(csc_matrix((t[1].data, (t[1].nonzero()[0], t[1].nonzero()[0])), shape=self.shape)).data)

        return res


class FnpFlp(object):
    def f(self, t):
        node, l_coefs, res, node.active = t[1], t[0], 0., False

        if len(l_coefs) > 0:
            if node.level > 0:
                if len(l_coefs) >= node.level:
                    res = node.encode(l_coefs)
                    node.active = True
            else:
                res = node.frequency_stack.encode(l_coefs)
                node.level = len(l_coefs)
                node.active = True

        return res, node.active

class FCP(object):
    def f(self, t):
        node, l_coefs, res, node.active = t[1], t[0], 0., False

        if len(l_coefs) > 0:
            if node.level > 0:
                if len(l_coefs) >= node.level:
                    res = node.encode(l_coefs)
                    node.active = True
            else:
                res = node.encode(l_coefs)
                node.level = len(l_coefs)

        return res


def fnt(sax_D, sax_I, sax_sn, sax_si, n_jobs=0):

    p = Pool({0: cpu_count()}.get(n_jobs, n_jobs))

    # Instantiate class that implement inner product
    fnt_ = FNT(shape=(sax_I.shape[0], sax_I.shape[0]))
    _res = p.map(fnt_.f, zip([sax_si] * sax_I.shape[-1], [sax_I[:, i] for i in range(sax_I.shape[-1])]))

    fnt_.shape = (sax_D.shape[0], sax_D.shape[0])
    res_ = p.map(fnt_.f, zip([sax_sn] * sax_D.shape[-1], [sax_D[:, i] for i in range(sax_D.shape[-1])]))

    return [_r + r_ for _r, r_ in zip(_res, res_)]


def fot(sax_O, sax_sn):
    sax_sn_ = csc_matrix((np.ones(len(sax_sn.data)), sax_sn.nonzero()), shape=sax_sn.shape)
    res = la.matrix_product(sax_sn_, sax_O)

    return res


def forward_processing(l_fnt, l_nodes, sax_Cm):
    """
    Update networks node, network signal at next iteratior and candidate matrix
    :param l_fnt: list work nodes of signal receive from previous forward transmitting phase
    :param l_nodes: list of network nodes
    :param sax_Cm: scipy.sparse.csc_matrix of historical connection between network nodes and output nodes
    :return:
    """
    l_res = []
    for l_coefs, node in zip(l_fnt, l_nodes):
        if len(l_coefs) > 0:
            if node.level > 0:
                if len(l_coefs) >= node.level:
                    l_res += [node.frequency_stack.encode(l_coefs)]

            else:
                l_res += [node.frequency_stack.encode(l_coefs)]
                node.level = len(l_coefs)

            node.active = True
        else:
            node.active = False

    # Build sn
    sax_sn = csc_matrix(l_res)

    # Build candidate matrix
    sax_C = csc_matrix(np.array([[n.active for n in l_nodes]]).repeat(sax_Cm.shape[1], axis=0).transpose())
    sax_C -= sax_Cm

    return sax_sn, csc_matrix((sax_C.data > 0, sax_C.nonzero()), shape=sax_C.shape)














# Global import
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool, cpu_count
from scipy.sparse import csc_matrix
# Local import
from deyep.core.tools import linear_algebra as la
from deyep.utils.names import KVName


def fnt(sax_D, sax_I, sax_sn, sax_si):
    res = la.matrix_product(sax_sn, sax_D) + la.matrix_product(sax_si, sax_I)
    return res


def fot(sax_O, sax_sn):
    res = la.matrix_product(sax_sn, sax_O)
    return res


def fnp(sax_fnt, l_nodes):
    sax_sn = csc_matrix(sax_fnt.shape)

    # Init activation param of the nodes
    for n in l_nodes:
        n.active = False
    # TODO: update level here
    # Encode forward messages
    for i in np.unique(sax_fnt.nonzero()[1]):
        s_out = l_nodes[i].frequency_stack.encode(sax_fnt[:, i].toarray()[:, 0])
        sax_sn[:, i] = np.array([s_out]).transpose()
        l_nodes[i].active = True

    return sax_sn


def fcp(l_actives, sax_Cm):
    # Build candidate matrix
    sax_C = csc_matrix(np.array([l_actives]).repeat(sax_Cm.shape[1], axis=0).transpose())
    sax_C -= sax_Cm

    return csc_matrix((sax_C.data > 0, sax_C.nonzero()), shape=sax_C.shape)


def bnt():
    raise NotImplementedError


def bit():
    raise NotImplementedError


def bop(sax_so, sax_got):

    sax_active = csc_matrix([sax_so[0, i] != 0 for i in range(sax_so.shape[-1])])

    no = sax_active.shape[1]
    sax_sob = ((2 * sax_got) - sax_active)\
        .dot(csc_matrix((sax_active.data, (sax_active.nonzero()[1], sax_active.nonzero()[1])), shape=(no, no)))
    sax_sob = sax_so.dot(csc_matrix((sax_sob.data, (sax_sob.nonzero()[1], sax_sob.nonzero()[1])), shape=(no, no)))

    return sax_sob













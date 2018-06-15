# Global import
import numpy as np
from scipy.sparse import csc_matrix
from pathos.multiprocessing import ProcessingPool as Pool, cpu_count

# Local import
from deyep.core.tools import linear_algebra as la


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

    # Encode forward messages, update list of activation and level if necessary
    for i in np.unique(sax_fnt.nonzero()[1]):
        s_out, level = l_nodes[i].frequency_stack.encode(sax_fnt[:, i].toarray()[:, 0], return_level=True)

        if level >= l_nodes[i].level:
            l_nodes[i].active = True
            sax_sn[:, i] = np.array([s_out]).transpose()

        if l_nodes[i].level == 0:
            l_nodes[i].level = level

    return sax_sn


def fcp(l_actives, sax_Cm):

    # Build candidate matrix
    sax_C = csc_matrix(np.array([l_actives]).repeat(sax_Cm.shape[1], axis=0).transpose())
    sax_C -= sax_Cm

    return csc_matrix((sax_C.data > 0, sax_C.indices, sax_C.indptr), shape=sax_C.shape).toarray()


def bnt(sax_D, sax_O, sax_snb, sax_sob, sax_activation):

    sax_snb_ = la.matrix_product(sax_sob, sax_O.transpose().multiply(sax_activation))

    sax_snb_ += la.matrix_product(sax_snb, sax_D.transpose())

    return sax_snb_


def bit(sax_I, sax_snb):
    sax_sib = la.matrix_product(sax_snb, sax_I.transpose())

    return sax_sib


def bop(sax_so, sax_got, N):

    # Get activated outputs and normalize signal
    sax_active = csc_matrix([sax_so[0, i] != 0 for i in range(sax_so.shape[-1])])

    # Init result
    sax_so_ = csc_matrix((sax_so.shape[1], sax_so.shape[0]))

    # Apply function chi (fourrier series domain) to output signal
    l_basis = [la.get_fourrier_coef_from_params(N, k) for k in range(N / 2)]
    for i in range(sax_so.shape[-1]):
        sax_so_[i, :] = la.Chi_fourrier(sax_so[:, i].toarray()[:, 0], l_basis,  n_jobs=0)

    # Compute feedback
    no = sax_active.shape[1]
    sax_sob = ((2 * sax_got) - sax_active)\
        .dot(csc_matrix((sax_active.data, (sax_active.nonzero()[1], sax_active.nonzero()[1])), shape=(no, no)))

    sax_sob = sax_so_.transpose().dot(csc_matrix((sax_sob.data, (sax_sob.nonzero()[1], sax_sob.nonzero()[1])), shape=(no, no)))

    return sax_sob


def bnp(l_nodes, sax_snb):

    sax_snb_ = csc_matrix((sax_snb.shape[1], sax_snb.shape[0]))
    for i in range(sax_snb.shape[1]):
        if sax_snb[:, i].nnz > 0:
            sax_snb_[i, :] = l_nodes[i].frequency_stack.decode(sax_snb.toarray()[:, i])

    return sax_snb_.transpose()


def bcu(sax_sob, sax_Cb, dn):
    # Simply follow the theoretical equation and update deep network's Cm and O

    raise NotImplementedError


class BduParallel(object):

    def f(self, t):
        res = 0
        for x in t[1].frequency_stack.fourrier_basis(free=False):
            res += np.round(np.real(la.inner_product(x, t[2])))

        return t[0], res


def bdu(sax_snb, Dw, l_nodes):

    # Decompose Dw
    l_indices = zip(*Dw.nonzero())

    p = Pool(cpu_count())

    # Instantiate class that implement inner product
    bdup = BduParallel()

    # Parallel inner product
    l_res = p.map(bdup.f, [((i, j), l_nodes[i], sax_snb[:, j].toarray()[:, 0]) for (i, j) in l_indices])

    # TODO: create update matrix D from l_re

















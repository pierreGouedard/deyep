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


def fnp(sax_fnt, l_nodes, tau):

    sax_sn = csc_matrix((sax_fnt.shape[1], sax_fnt.shape[0]), dtype=complex)

    # Init activation param of the nodes
    for n in l_nodes:
        n.active = False

    # Encode forward messages, update list of activation and level if necessary
    for i in np.unique(sax_fnt.nonzero()[1]):
        s_out, level = l_nodes[i].frequency_stack.encode(sax_fnt[:, i].toarray()[:, 0], return_level=True)

        if l_nodes[i].d_levels[level] >= tau:
            l_nodes[i].active = True
            sax_sn[i, :] = s_out

    return sax_sn.transpose(), np.array([n.active for n in l_nodes])


def fcp(l_actives, sax_Cm):

    # Build candidate matrix
    sax_C = csc_matrix(np.array([l_actives]).repeat(sax_Cm.shape[1], axis=0).transpose())
    sax_C -= sax_Cm

    return csc_matrix((sax_C.data > 0, sax_C.indices, sax_C.indptr), shape=sax_C.shape)


def buffer(sax_C, ax_sa, no, sax_so):
    return sax_C.copy(), csc_matrix(np.array([ax_sa]).repeat(no, axis=0)), sax_so.copy()


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
    sax_so_ = csc_matrix((sax_so.shape[1], sax_so.shape[0]), dtype=complex)

    # Apply function chi (fourrier series domain) to output signal
    l_basis = [la.get_fourrier_coef_from_params(N, k) for k in range(N / 2)]

    for i in np.unique(sax_so.nonzero()[1]):
        sax_so_[i, :] = la.Chi_fourrier(sax_so[:, i].toarray()[:, 0], l_basis,  n_jobs=0)

    # Compute feedback
    no = sax_active.shape[1]
    sax_sob = ((2 * sax_got) - sax_active)\
        .dot(csc_matrix((sax_active.data, (sax_active.nonzero()[1], sax_active.nonzero()[1])), shape=(no, no)))

    sax_sob = sax_so_.transpose().dot(csc_matrix((sax_sob.data, (sax_sob.nonzero()[1], sax_sob.nonzero()[1])), shape=(no, no)))

    return sax_sob


def bnp(l_nodes, sax_snb):

    sax_snb_ = csc_matrix((sax_snb.shape[1], sax_snb.shape[0]), dtype=complex)
    for i in np.unique(sax_snb.nonzero()[1]):
        s, d_levels = l_nodes[i].frequency_stack.decode(sax_snb.toarray()[:, i], d_levels={})

        # Update Levels and backward signal
        for k, v in d_levels.items():
            l_nodes[i].d_levels[k] += v

        sax_snb_[i, :] = s

    return sax_snb_.transpose()


def bcu(sax_got, sax_Cb, dn, w0=1):

    for i in range(sax_got.shape[1]):
        sax_Cb[:, i] = sax_Cb[:, i] * sax_got[0, i] > 0

    # Update candidate memory and output edges
    dn.graph['Cm'] += sax_Cb
    dn.graph['Ow'] += sax_Cb * w0


class BduParallel(object):

    def f(self, t):
        res = 0
        for x in t[1].frequency_stack.fourrier_basis(free=False):
            res += np.round(np.real(la.inner_product(x, t[2])))

        return t[0], res


def bdu(sax_snb, dn, penalty=1.):

    # Decompose Dw
    l_indices = zip(*dn.Dw.nonzero())
    l_nonzeros = sax_snb.nonzero()[1]
    l_indices = [(i, j) for i, j in l_indices if j in l_nonzeros]

    p = Pool(cpu_count())

    # Instantiate class that implement inner product
    bdup = BduParallel()

    # Parallel inner product
    l_res = p.map(bdup.f, [((i, j), dn.network_nodes[i], sax_snb[:, j].toarray()[:, 0]) for (i, j) in l_indices])

    Du = csc_matrix(dn.Dw.shape)
    for (i, j), x in filter(lambda x: x != 0, l_res):
        Du[i, j] = min(x * penalty, 1.)

    dn.graph['Dw'] += Du


class BouParallel(object):

    def f(self, t):
        res = 0
        for x in t[1].frequency_stack.fourrier_basis(free=False):
            res += np.round(np.real(la.inner_product(x, t[2])))

        return t[0], res


def bou(sax_sob, sax_activation, dn, penalty=1.):

    # Decompose Dw
    l_indices = zip(*dn.Ow.multiply(sax_activation.transpose()).nonzero())
    l_nonzeros = sax_sob.nonzero()[1]
    l_indices = [(i, j) for i, j in l_indices if j in l_nonzeros]

    p = Pool(cpu_count())

    # Instantiate class that implement inner product
    boup = BouParallel()

    # Parallel inner product
    l_res = p.map(boup.f, [((i, j), dn.network_nodes[i], sax_sob[:, j].toarray()[:, 0]) for (i, j) in l_indices])

    Ou = csc_matrix(dn.Ow.shape)

    for (i, j), x in filter(lambda (_, y): y != 0, l_res):
        Ou[i, j] = min(x * penalty, 1.)

    dn.graph['Ow'] += Ou


class BiuParallel(object):

    def f(self, t):
        res = 0
        for x in t[1].frequency_stack.fourrier_basis(free=True):
            res += np.round(np.real(la.inner_product(x, t[2])))

        return t[0], res


def biu(sax_snb, dn, penalty=1.):

    # Decompose Iw
    l_indices = zip(*dn.Iw.nonzero())
    l_nonzeros = sax_snb.nonzero()[1]
    l_indices = [(i, j) for i, j in l_indices if j in l_nonzeros]

    p = Pool(cpu_count())

    # Instantiate class that implement inner product
    biup = BiuParallel()

    # Parallel inner product
    l_res = p.map(biup.f, [((i, j), dn.input_nodes[i], sax_snb[:, j].toarray()[:, 0]) for (i, j) in l_indices])
    Iu = csc_matrix(dn.Iw.shape)

    for (i, j), x in filter(lambda (_, y): y != 0, l_res):
        Iu[i, j] = min(x * penalty, 1.)

    dn.graph['Iw'] += Iu

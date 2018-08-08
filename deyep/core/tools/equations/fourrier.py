# Global import
import numpy as np
from scipy.sparse import csc_matrix
from pathos.multiprocessing import ProcessingPool as Pool, cpu_count

# Local import
from deyep.core.tools.linear_algebra.comon import Chi_ax


def f_fnp(sax_fnt, l_nodes, tau, t):
    sax_sn = csc_matrix((sax_fnt.shape[1], sax_fnt.shape[0]), dtype=complex)

    # Init activation param of the nodes
    for n in l_nodes:
        n.active = False

    # Encode forward messages, update list of activation and level if necessary
    for i in np.unique(sax_fnt.nonzero()[1]):
        s_out, level = l_nodes[i].basis.encode(sax_fnt[:, i].toarray()[:, 0], timestamp=t, return_level=True)
        if l_nodes[i].d_levels[level] >= tau:
            l_nodes[i].active = True
            sax_sn[i, :] = s_out

    return sax_sn.transpose(), np.array([n.active for n in l_nodes])


def f_fcp(l_actives, sax_Cm):

    # Build candidate matrix
    sax_C = csc_matrix(np.array([l_actives]).repeat(sax_Cm.shape[1], axis=0).transpose())
    sax_C -= sax_Cm

    return csc_matrix((sax_C.data > 0, sax_C.indices, sax_C.indptr), shape=sax_C.shape)


def f_buffer(sax_C, ax_sa, no, sax_so):
    return sax_C.copy(), csc_matrix(np.array([ax_sa.copy()]).repeat(no, axis=0)), sax_so.copy()


class BopParallel(object):
    def __init__(self, ax_basis):
        self.ax_basis = ax_basis

    def f(self, t):
        ax_coef = Chi_ax(np.round(np.real(t[1].dot(self.ax_basis.conjugate()))))
        return t[0], ax_coef.dot(self.ax_basis.transpose())


def f_bop(sax_so, sax_got, l_active_nodes, n_jobs=0):

    # Apply function chi (fourrier series domain) to output signal
    l_basis, l_keys = [], []
    for node in l_active_nodes:
        if node.basis.key not in l_keys:
            l_keys += [node.basis.key]
            l_basis += [node.basis.base +
                        node.basis.base_from_key('N={},k={}'.format(node.basis.N, node.basis.key), offset=1)]

    if len(l_basis) > 0:

        p = Pool({0: cpu_count()}.get(n_jobs, n_jobs))

        # Instantiate class that implement parallel operations
        bopp = BopParallel(np.vstack(tuple(l_basis)).transpose())

        # Parallel operations
        l_ins = [(i, sax_so.toarray()[:, i]) for i in np.unique(sax_so.nonzero()[1])]
        l_res = p.map(bopp.f, l_ins)

        # Fill  sax_so_
        sax_so = csc_matrix((sax_so.shape[1], sax_so.shape[0]), dtype=complex)
        for i, ax_s in l_res:
            sax_so[i, :] = ax_s
    else:
        sax_so = sax_so.transpose()

    # Compute feedback
    l_sio = [sax_so[i, 0] != 0 for i in range(sax_so.shape[0])]
    sax_sob = ((2 * sax_got) - csc_matrix(l_sio)).dot(csc_matrix(np.diag(l_sio)))
    sax_sob = csc_matrix((sax_sob.data, (sax_sob.nonzero()[1], sax_sob.nonzero()[1])), shape=(len(l_sio), len(l_sio)))

    return sax_sob.dot(sax_so).transpose()


class BnpParallel(object):
    def __init__(self, t, l_key_inputs):
        self.t = t
        self.l_key_inputs = l_key_inputs

    def f(self, t):
        res = t[1].basis.decode(t[2], self.t, self.l_key_inputs, d_levels={})
        return t[0], res[0], res[1]


def f_bnp(l_nodes, sax_snb, t, l_key_inputs, n_jobs=0):
    p = Pool({0: cpu_count()}.get(n_jobs, n_jobs))

    # Instantiate class that implement parallel operations
    bnpp = BnpParallel(t, l_key_inputs)

    # Parallel operations
    l_ins = [(i, l_nodes[i], sax_snb.toarray()[:, i]) for i in np.unique(sax_snb.nonzero()[1])]
    l_res = filter(lambda x: x is not None, p.map(bnpp.f, l_ins))

    # Fill
    sax_snb = csc_matrix((sax_snb.shape[1], sax_snb.shape[0]), dtype=complex)
    for i, s, d_levels in l_res:
        # Update Levels and backward signal
        for k, v in d_levels.items():
            l_nodes[i].d_levels[k] += v

        sax_snb[i, :] = s

    return sax_snb.transpose()


class BduParallel(object):
    @staticmethod
    def f(t):
        res = np.round(np.real(t[1].basis.base.dot(t[2].conjugate())))
        return t[0], res


def f_bdu(sax_snb, dn, penalty=1.):

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
    @staticmethod
    def f(t):
        res = np.round(np.real(t[1].basis.base.dot(t[2].conjugate())))
        return t[0], res


def f_bou(sax_sob, sax_activation, dn, penalty=1.):

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
    @staticmethod
    def f(t):
        res = np.round(np.real(t[1].basis.base.dot(t[2].conjugate())))
        return t[0], res


def f_biu(sax_snb, dn, penalty=1.):

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

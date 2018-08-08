# Global import
import numpy as np
from scipy.sparse import csc_matrix, vstack
from pathos.multiprocessing import ProcessingPool as Pool, cpu_count

# Local import
from deyep.core.tools.linear_algebra.comon import Chi_sax


def c_fnp(sax_fnt, l_nodes, tau, t):
    sax_sn = csc_matrix((sax_fnt.shape[1], sax_fnt.shape[0]), dtype=int)

    # Init activation param of the nodes
    for n in l_nodes:
        n.active = False

    # Encode forward messages, update list of activation and level if necessary
    for i in np.unique(sax_fnt.nonzero()[1]):
        s_out, level = l_nodes[i].basis.encode(sax_fnt[:, i].transpose(), timestamp=t, return_level=True)
        if l_nodes[i].d_levels[level] >= tau:
            l_nodes[i].active = True
            sax_sn[i, :] = s_out

    return sax_sn.transpose(), np.array([n.active for n in l_nodes])


def c_fcp(l_actives, sax_Cm):

    # Build candidate matrix
    sax_C = csc_matrix(np.array([l_actives]).repeat(sax_Cm.shape[1], axis=0).transpose())
    sax_C -= sax_Cm

    return csc_matrix((sax_C.data > 0, sax_C.indices, sax_C.indptr), shape=sax_C.shape)


def c_buffer(sax_C, ax_sa, no, sax_so):
    return sax_C.copy(), csc_matrix(np.array([ax_sa.copy()]).repeat(no, axis=0)), sax_so.copy()


def c_bop(sax_so, sax_got):
    sax_so = Chi_sax(vstack([csc_matrix((1, sax_so.shape[1])), sax_so[:-1, :]]) + sax_so).transpose()

    # Compute feedback TODO: may be optimized
    sax_sio = Chi_sax(csc_matrix(sax_so.sum(axis=1)).transpose())
    sax_sob = ((2 * sax_got) - sax_sio).dot(csc_matrix(np.diag(sax_sio.toarray()[0])))
    sax_sob = csc_matrix(np.diag(sax_sob.toarray()[0]))

    return sax_sob.dot(sax_so).transpose()


class BnpParallel(object):
    def __init__(self, t, l_key_inputs):
        self.t = t
        self.l_key_inputs = l_key_inputs

    def f(self, t):
        res = t[1].basis.decode(t[2], self.t, self.l_key_inputs, d_levels={})
        return t[0], res[0], res[1]


def c_bnp(l_nodes, sax_snb, t, l_key_inputs, n_jobs=0):
    p = Pool({0: cpu_count()}.get(n_jobs, n_jobs))

    # Instantiate class that implement parallel operations
    bnpp = BnpParallel(t, l_key_inputs)

    # Parallel operations
    l_ins = [(i, l_nodes[i], sax_snb[:, i].transpose()) for i in np.unique(sax_snb.nonzero()[1])]
    l_res = filter(lambda x: x is not None, p.map(bnpp.f, l_ins))

    # Fill
    sax_snb = csc_matrix((sax_snb.shape[1], sax_snb.shape[0]), dtype=int)
    for i, s, d_levels in l_res:
        # Update Levels and backward signal
        for k, v in d_levels.items():
            l_nodes[i].d_levels[k] += v

        sax_snb[i, :] = s

    return sax_snb.transpose()


class ParallelUpdate(object):
    def __init__(self, sax_bw):
        self.sax_bw = sax_bw

    def f(self, t):
        sax_res = t[0].dot(self.sax_bw).multiply(t[1])
        return sax_res


def c_bdu(sax_snb, dn, penalty=1., n_jobs=1):

    if penalty != 1.:
        sax_snb = (sax_snb < 0) * penalty + (sax_snb > 0)

    if n_jobs != 1:
        p = Pool(cpu_count())

        # Instantiate class that implement inner product
        bdup = ParallelUpdate(sax_snb)

        # Parallel inner product (map from Pool preserve order of input list)
        sax_Du = vstack(p.map(bdup.f, [(n.basis.base, dn.D[i, :]) for i, n in enumerate(dn.network_nodes)]))
    else:
        sax_Du = vstack([n.basis.base for n in dn.network_nodes]).dot(sax_snb).multiply(dn.D)

    dn.graph['Dw'] += sax_Du


def c_bou(sax_sob, sax_A, dn, penalty=1., n_jobs=1):

    if penalty != 1.:
        sax_sob = (sax_sob < 0) * penalty + (sax_sob > 0)

    if n_jobs != 1:
        p = Pool(cpu_count())

        # Instantiate class that implement inner product
        boup = ParallelUpdate(sax_sob)

        # Parallel inner product (map from Pool preserve order of input list)
        sax_Ou = vstack(p.map(boup.f, [(n.basis.base, dn.O[i, :].multiply(sax_A[:, i].transpose()))
                                       for i, n in enumerate(dn.network_nodes)]))
    else:
        sax_Ou = vstack([n.basis.base for n in dn.network_nodes])\
            .dot(sax_sob)\
            .multiply(dn.O.multiply(sax_A.transpose()))

    dn.graph['Ow'] += sax_Ou


def c_biu(sax_snb, dn, penalty=1., n_jobs=1):

    if penalty != 1.:
        sax_snb = (sax_snb < 0) * penalty + (sax_snb > 0)

    if n_jobs != 1:
        p = Pool(cpu_count())

        # Instantiate class that implement inner product
        biup = ParallelUpdate(sax_snb)

        # Parallel inner product (map from Pool preserve order of input list)
        sax_Iu = vstack(p.map(biup.f, [(n.basis.base, dn.I[i, :]) for i, n in enumerate(dn.input_nodes)]))
    else:
        sax_Iu = vstack([n.basis.base for n in dn.input_nodes]).dot(sax_snb).multiply(dn.I)

    dn.graph['Iw'] += sax_Iu


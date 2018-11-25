# Global import
import numpy as np
from scipy.sparse import csc_matrix, vstack, lil_matrix
from pathos.multiprocessing import ProcessingPool as Pool, cpu_count

# Local import
from deyep.core.tools.linear_algebra.comon import Chi_sax


def bnt(sax_D, sax_O, sax_snb, sax_sob, sax_activation):
    sax_snb_ = sax_sob.dot(sax_O.transpose().multiply(sax_activation))
    sax_snb_ += sax_snb.dot(sax_D.transpose())
    return sax_snb_


def bit(sax_I, sax_snb):
    sax_sib = sax_snb.dot(sax_I.transpose())
    return sax_sib


def bcv(sax_got, sax_sob, sax_Cb):
    for i in range(sax_got.shape[1]):
        sax_Cb[:, i] = sax_Cb[:, i] * (sax_got[0, i] - sax_sob[0, i]) > 0

    return sax_Cb


def buffer(sax_C, ax_sa, no, sax_so):
    return sax_C.copy(), csc_matrix(np.array([ax_sa.copy()]).repeat(no, axis=0)), sax_so.copy()


def bop(sax_so, sax_got):

    sax_so = Chi_sax(vstack([csc_matrix((1, sax_so.shape[1])), sax_so[:-1, :]], format='csc') + sax_so).transpose()

    # Compute feedback TODO: may be optimized
    sax_sio = Chi_sax(csc_matrix(sax_so.sum(axis=1)).transpose())
    sax_sob = ((2 * sax_got) - sax_sio).dot(csc_matrix(np.diag(sax_sio.toarray()[0])))
    sax_sob = csc_matrix(np.diag(sax_sob.toarray()[0]))

    return sax_sob.dot(sax_so).transpose()


class BnpParallel(object):
    def __init__(self, t, key_inputs):
        self.t = t
        self.key_inputs = key_inputs

    def f(self, t):
        res = t[1].basis.decode(t[2], self.t, self.key_inputs, d_levels={})

        if res is None:
            return res

        return t[0], res[0], res[1]


def bnp(l_nodes, sax_snb, t, key_inputs, n_jobs=0):
    p = Pool({0: cpu_count()}.get(n_jobs, n_jobs))

    # Instantiate class that implement parallel operations
    bnpp = BnpParallel(t, key_inputs)

    # Parallel operations
    l_ins = [(i, l_nodes[i], sax_snb[:, i].transpose()) for i in np.unique(sax_snb.nonzero()[1])]
    l_res = filter(lambda x: x is not None, p.map(bnpp.f, l_ins))

    # Fill
    sax_snb = lil_matrix((sax_snb.shape[1], sax_snb.shape[0]), dtype=int)
    for i, s, d_levels in l_res:
        # Update Levels and backward signal
        for k, v in d_levels.items():
            l_nodes[i].d_levels[k] += v
        sax_snb[i, :] = s

    return sax_snb.tocsc().transpose()


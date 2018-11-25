# Global import
import numpy as np
from scipy.sparse import vstack
from pathos.multiprocessing import ProcessingPool as Pool, cpu_count

# Local import


class ParallelUpdate(object):
    def __init__(self, sax_bw):
        self.sax_bw = sax_bw

    def f(self, t):
        sax_res = t[0].dot(self.sax_bw).multiply(t[1])
        return sax_res


def bcu(sax_Cb, dn, w0=1):
    dn.graph['Cm'] += sax_Cb
    dn.graph['Ow'] += sax_Cb * w0


def bdu(sax_snb, dn, penalty=1., n_jobs=0):

    sax_snb_ = (penalty * sax_snb < 0).multiply(sax_snb) + (sax_snb > 0).multiply(sax_snb)

    if n_jobs != 1:
        p = Pool(cpu_count())

        # Instantiate class that implement inner product
        bdup = ParallelUpdate(sax_snb_)

        # Parallel inner product (map from Pool preserve order of input list)
        sax_Du = vstack(p.map(bdup.f, [(n.basis.base, dn.D[i, :]) for i, n in enumerate(dn.network_nodes)]), format='csc')
    else:
        sax_Du = vstack([n.basis.base for n in dn.network_nodes], format='csc').dot(sax_snb_).multiply(dn.D)

    ax_count = np.array(sax_Du.sum(axis=1), dtype=int)

    if penalty != 1.:
        ax_count /= penalty

    dn.graph['Dw'] += sax_Du

    return ax_count[:, 0]


def bou(sax_sob, sax_A, dn, penalty=1., n_jobs=0):

    if penalty != 1. and (sax_sob < 0).nnz > 0:
        sax_sob[sax_sob < 0] = sax_sob[sax_sob < 0] * penalty

    if n_jobs != 1:
        p = Pool(cpu_count())

        # Instantiate class that implement inner product
        boup = ParallelUpdate(sax_sob)

        # Parallel inner product (map from Pool preserve order of input list)
        sax_Ou = vstack(p.map(boup.f, [(n.basis.base, dn.O[i, :].multiply(sax_A[:, i].transpose()))
                                       for i, n in enumerate(dn.network_nodes)]), format='csc')
    else:
        sax_Ou = vstack([n.basis.base for n in dn.network_nodes], format='csc')\
            .dot(sax_sob)\
            .multiply(dn.O.multiply(sax_A.transpose()))

    ax_count = np.array(sax_Ou.sum(axis=1), dtype=int)

    if penalty != 1.:
        ax_count /= penalty

    dn.graph['Ow'] += sax_Ou

    return ax_count[:, 0]


def biu(sax_snb, dn, penalty=1., n_jobs=0):

    if penalty != 1.:
        sax_snb = ((sax_snb < 0) * -1 * penalty) + (sax_snb > 0)

    if n_jobs != 1:
        p = Pool(cpu_count())

        # Instantiate class that implement inner product
        biup = ParallelUpdate(sax_snb)

        # Parallel inner product (map from Pool preserve order of input list)
        sax_Iu = vstack(p.map(biup.f, [(n.basis.base, dn.I[i, :]) for i, n in enumerate(dn.input_nodes)]), format='csc')
    else:
        sax_Iu = vstack([n.basis.base for n in dn.input_nodes], format='csc').dot(sax_snb).multiply(dn.I)

    dn.graph['Iw'] += sax_Iu

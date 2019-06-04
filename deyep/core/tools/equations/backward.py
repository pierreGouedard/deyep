# Global import
import numpy as np
from scipy.sparse import csc_matrix, vstack, lil_matrix
from pathos.multiprocessing import ProcessingPool as Pool, cpu_count

# Local import
from deyep.core.tools.linear_algebra.comon import Chi_sax


def bnt(sax_D, sax_O, sax_snb, sax_sob, sax_activation):
    """
    Propagate backward signal through firing graph

    :param sax_D: scipy.sparse matrices of direct connection of core vertices
    :param sax_O: scipy.sparse matrices of direct connection of core vertices toward output vertices
    :param sax_snb: scipy.sparse of backward signals of core vertices
    :param sax_sob: scipy.sparse of backward signals of output vertices
    :param sax_activation: scipy.sparse array of activation of core vertices
    :return: scipy.sparse of backward signals received by core vertices

    """
    sax_snb_ = sax_sob.dot(sax_O.transpose().multiply(sax_activation))
    sax_snb_ += sax_snb.dot(sax_D.transpose())
    return sax_snb_


def bit(sax_I, sax_snb):
    """
    Propagate backward signal to input vertices if the firing graph

    :param sax_I: scipy.sparse matrices of direct connection of input vertices toward core vertices
    :param sax_snb: scipy.sparse of backward signals of core vertices
    :return: scipy.sparse of backward signals received by input vertices
    """
    sax_sib = sax_snb.dot(sax_I.transpose())
    return sax_sib


def buffer(ax_sa, no, sax_so):
    """
    Buffer forward signals before backward processing and transmitting

    :param ax_sa: numpy.array of active vertex
    :param no: number of output vertex
    :param sax_so: scipy.sparse forward signal received by output vertices
    :return: Copy of the input

    """
    return csc_matrix(np.array([ax_sa.copy()]).repeat(no, axis=0)), sax_so.copy()


def bop(sax_so, sax_got):
    """
    Backward processing of signals of output vertices

    :param sax_so: scipy.sparse forward  signals received by output vertices
    :param sax_got: scipy.sparse Ground of truth of signals for output vertices
    :return:
    """

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
        res = t[1].basis.decode(t[2], self.t, self.key_inputs)
        return t[0], res


def bnp(l_vertices, sax_snb, t, key_inputs, n_jobs=0):
    """
    Backward processing of signals of core vertices

    :param l_vertices: list of core vertices
    :param sax_snb: scipy.sparse of backward signals of core vertices
    :param t: int timestamp
    :param key_inputs: input's vertices frequency keys
    :param n_jobs: int core used, if 0: use all available core
    :return:
    """
    p = Pool({0: cpu_count()}.get(n_jobs, n_jobs))

    # Instantiate class that implement parallel operations
    bnpp = BnpParallel(t, key_inputs)

    # Parallel operations
    l_ins = [(i, l_vertices[i], sax_snb[:, i].transpose()) for i in np.unique(sax_snb.nonzero()[1])]
    #l_res = filter(lambda x: x is not None, p.map(bnpp.f, l_ins))

    l_res = []
    for tu in l_ins:
        l_res += [(tu[0], tu[1].basis.decode(tu[2], t, key_inputs))]

    # Fill
    # TODO: change that fuck

    sax_snb = lil_matrix((sax_snb.shape[1], sax_snb.shape[0]), dtype=int)
    for i, s in l_res:
        sax_snb[i, :] = s

    return sax_snb.tocsc().transpose()


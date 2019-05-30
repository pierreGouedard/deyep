# Global import
import numpy as np
from scipy.sparse import lil_matrix

# Local import


def fnt(sax_D, sax_I, sax_sn, sax_si):
    """
    Transmit signal through core vertices of firing graph

    :param sax_D: scipy.sparse matrices of direct connection of core vertices
    :param sax_I: scipy.sparse matrices of direct connection of input vertices toward core vertices
    :param sax_sn: scipy.sparse signals of core vertices
    :param sax_si: scipy.sparse input signals of input vertices
    :return: scipy.sparse output signals of core vertices
    """
    res = sax_sn.dot(sax_D) + sax_si.dot(sax_I)
    return res


def fot(sax_O, sax_sn):
    """
    Transmit signal of core vertices to output vertices of firing graph

    :param sax_O: scipy.sparse matrices of direct connection of core vertices toward output vertices
    :param sax_sn: scipy.sparse signals of core vertices
    :return: scipy.sparse output signals of output vertices
    """
    res = sax_sn.dot(sax_O)
    return res


def fnp(sax_fnt, l_core_vertices, t):
    """
    core vertex processing of received signals

    :param sax_fnt: scipy.sparse received signals
    :param l_core_vertices: list of core vertex of firing graph
    :param t: int timestamp
    :return: scipy.sparse processed signals
    """

    sax_sn = lil_matrix((sax_fnt.shape[1], sax_fnt.shape[0]), dtype=int)

    # Init activation param of the nodes
    for n in l_core_vertices:
        n.active = False

    # Encode forward messages, update list of activation
    for i in np.unique(sax_fnt.nonzero()[1]):
        s_out = l_core_vertices[i].basis.encode(sax_fnt[:, i].transpose(), t, l_core_vertices[i].l0)
        if s_out is not None:
            l_core_vertices[i].active = True
            sax_sn[i, :] = s_out

    return sax_sn.tocsc().transpose(), np.array([n.active for n in l_core_vertices])
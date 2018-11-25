# Global import
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix

# Local import


def fnt(sax_D, sax_I, sax_sn, sax_si):
    res = sax_sn.dot(sax_D) + sax_si.dot(sax_I)
    return res


def fot(sax_O, sax_sn):
    res = sax_sn.dot(sax_O)
    return res


def fnp(sax_fnt, l_nodes, tau, t):
    sax_sn = lil_matrix((sax_fnt.shape[1], sax_fnt.shape[0]), dtype=int)

    # Init activation param of the nodes
    for n in l_nodes:
        n.active = False

    # Encode forward messages, update list of activation and level if necessary
    for i in np.unique(sax_fnt.nonzero()[1]):
        s_out, level = l_nodes[i].basis.encode(sax_fnt[:, i].transpose(), timestamp=t, return_level=True)
        if l_nodes[i].d_levels[level] >= tau:
            l_nodes[i].active = True
            sax_sn[i, :] = s_out

    return sax_sn.tocsc().transpose(), np.array([n.active for n in l_nodes])


def fcp(l_actives, sax_Cm):
    # Build candidate matrix
    sax_C = csc_matrix(np.array([l_actives]).repeat(sax_Cm.shape[1], axis=0).transpose())

    return (sax_C.astype(int) - sax_Cm.astype(int)) > 0


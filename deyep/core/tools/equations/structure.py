# Global import
from scipy.sparse import vstack
from pathos.multiprocessing import ProcessingPool as Pool, cpu_count

# Local import


# TODO: Fix Problem with parallelisation (very uneficient)
class ParallelUpdate(object):
    def __init__(self, sax_bw):
        self.sax_bw = sax_bw

    def f(self, t):
        sax_res = t[0].dot(self.sax_bw).multiply(t[1])
        return sax_res


# TODO add an option to decide which part of the structure are allowed to be updated
def bdu(sax_snb, fg, penalty=1., n_jobs=1):
    """
    Update structure of direct connection of core vertices of the graph with core backward signal

    :param sax_snb: scipy.sparse of backward signals of core vertices
    :param fg: deyep.core.firing_graph.FiringGraph
    :param penalty: float penalty
    :param n_jobs: int core used, if different from 1: use all available core
    :return: scipy.sparse update structure matrix
    """
    sax_snb_ = (penalty * sax_snb < 0).multiply(sax_snb) + (sax_snb > 0).multiply(sax_snb)

    if n_jobs != 1:
        p = Pool(cpu_count())

        # Instantiate class that implement inner product
        bdup = ParallelUpdate(sax_snb_)

        # Parallel inner product (map from Pool preserve order of input list)
        sax_Du = vstack(p.map(bdup.f, [(n.basis.base, fg.D[i, :]) for i, n in enumerate(fg.core_vertices)]), format='csc')
    else:
        sax_Du = vstack([n.basis.base for n in fg.network_nodes], format='csc').dot(sax_snb_).multiply(fg.D)

    fg.graph['Dw'] += sax_Du

    return sax_Du


def bou(sax_sob, sax_A, fg, penalty=1., n_jobs=1):
    """
    Update structure of the direct connection of core vertices toward output vertices
    using output backward signal

    :param sax_sob: scipy.sparse matrix of backward signals of output vertices
    :param sax_A: scipy.sparse matrix of active core vertices
    :param fg: deyep.core.firing_graph.FiringGraph
    :param penalty: int penalty
    :param n_jobs: int core used, if different from 1: use all available core
    :return: scipy.sparse update structure matrix
    """

    if penalty != 1. and (sax_sob < 0).nnz > 0:
        sax_sob[sax_sob < 0] = sax_sob[sax_sob < 0] * penalty

    if n_jobs != 1:
        p = Pool(cpu_count())

        # Instantiate class that implement inner product
        boup = ParallelUpdate(sax_sob)

        # Parallel inner product (map from Pool preserve order of input list)
        sax_Ou = vstack(p.map(boup.f, [(n.basis.base, fg.O[i, :].multiply(sax_A[:, i].transpose()))
                                       for i, n in enumerate(fg.network_nodes)]), format='csc')
    else:
        sax_Ou = vstack([n.basis.base for n in fg.network_nodes], format='csc')\
            .dot(sax_sob)\
            .multiply(fg.O.multiply(sax_A.transpose()))

    fg.graph['Ow'] += sax_Ou

    return sax_Ou


def biu(sax_snb, fg, penalty=1., n_jobs=1):
    """
    Update structure of the direct connection of input vertices toward core vertices
    using core backward signal

    :param sax_snb: scipy.sparse matrix of backward signals of core vertices
    :param fg: deyep.core.firing_graph.FiringGraph
    :param penalty: int penalty
    :param n_jobs: int core used, if different from 1: use all available core
    :return: scipy.sparse update structure matrix
    """

    if penalty != 1.:
        sax_snb = ((sax_snb < 0) * -1 * penalty) + (sax_snb > 0)

    if n_jobs != 1:
        p = Pool(cpu_count())

        # Instantiate class that implement inner product
        biup = ParallelUpdate(sax_snb)

        # Parallel inner product (map from Pool preserve order of input list)
        sax_Iu = vstack(p.map(biup.f, [(n.basis.base, fg.I[i, :]) for i, n in enumerate(fg.input_nodes)]), format='csc')
    else:
        sax_Iu = vstack([n.basis.base for n in fg.input_nodes], format='csc').dot(sax_snb).multiply(fg.I)

    fg.graph['Iw'] += sax_Iu

    return sax_Iu

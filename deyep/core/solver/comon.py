# Global import
from scipy.sparse import csc_matrix
import numpy as np

# Local import


class DeepNetSolver(object):
    def __init__(self, deep_network, delay, imputer, p0, basis, verbose=0):

        # Core params
        self.p = p0
        self.delay = delay
        self.basis = basis
        self.verbose = verbose

        # Data structure
        self.deep_network = deep_network
        self.imputer = imputer
        self.key_inputs = set(['N={},k={}'.format(n.basis.N, n.basis.key) for n in self.deep_network.input_nodes])

        # Init signals
        self.dtype = int if basis == 'canonical' else complex
        self.sax_si, self.sax_sn, self.sax_so, self.ax_sa, self.sax_C = init_core_forward_signal(deep_network, self.dtype)
        self.sax_sib, self.sax_snb, self.sax_sob, self.sax_sab, self.sax_Cb = init_core_forward_signal(deep_network, self.dtype)

        self.t = 0
        self.t_fp = 0

    def run_epoch(self, n):
        raise NotImplementedError

    def epoch_analysis(self, n):
        raise NotImplementedError

    def forward_transmiting(self):
        raise NotImplementedError

    def f_forward_processing(self, t):
        raise NotImplementedError

    def backward_transmiting(self, only_buffer=False):
        raise NotImplementedError

    def backward_processing(self, sax_got, t):
        raise NotImplementedError

    @staticmethod
    def generate_input_signals(sax_i, input_nodes, dtype):
        sax_si = csc_matrix((len(input_nodes), input_nodes[0].basis.N), dtype=dtype)

        for i,  n in enumerate(input_nodes):
            if sax_i[0, i] >= 1:
                sax_si[i, :] = n.basis.base

        return sax_si.transpose()


def init_core_forward_signal(dn, dtype):

    N = dn.n_freq
    sax_si = csc_matrix((N, len(dn.input_nodes)), dtype=dtype)
    sax_sn = csc_matrix((N, len(dn.network_nodes)), dtype=dtype)
    sax_so = csc_matrix((N, len(dn.output_nodes)), dtype=dtype)
    ax_sa = np.array([False] * len(dn.network_nodes))
    sax_C = csc_matrix((len(dn.network_nodes), len(dn.output_nodes)))

    return sax_si, sax_sn, sax_so, ax_sa, sax_C


def init_core_backward_signal(dn):
    N = dn.n_freq

    sax_sib = csc_matrix((N, len(dn.input_nodes)), dtype=dtype)
    sax_snb = csc_matrix((N, len(dn.network_nodes)), dtype=dtype)
    sax_sob = csc_matrix((N, len(dn.output_nodes)), dtype=dtype)
    sax_sab = csc_matrix(np.array([[False] * len(dn.network_nodes)]).repeat(len(dn.output_nodes), axis=0))
    sax_Cb = csc_matrix((len(dn.network_nodes), len(dn.output_nodes)))

    return sax_sib, sax_snb, sax_sob, sax_sab, sax_Cb

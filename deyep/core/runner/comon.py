# Global import
from scipy.sparse import csc_matrix
import numpy as np

# Local import


class DeepNetRunner(object):
    def __init__(self, deep_network, delay, imputer, basis=None, dry=False, verbose=0):
        # TODO here we need a deep_network dry option
        # Core params
        self.delay = delay
        self.basis = basis
        self.dry = dry
        self.verbose = verbose

        # Data structure
        self.deep_network = deep_network
        self.imputer = imputer

        # Init signals
        self.dtype = int if basis == 'canonical' else complex
        self.sax_si, self.sax_sn, self.sax_so, self.ax_sa, _ = init_core_forward_signal(deep_network, self.dtype)
        self.sax_sib, self.sax_snb, self.sax_sob, _, _ = init_core_backward_signal(deep_network, self.dtype)

    @staticmethod
    def generate_input_signals(sax_si, input_nodes, dtype):
        sax_si_ = csc_matrix((len(input_nodes), input_nodes[0].basis.N), dtype=dtype)

        for i,  n in enumerate(input_nodes):
            if sax_si[0, i] >= 1:
                sax_si_[i, :] = n.basis.base

        return sax_si_.transpose()

    @staticmethod
    def generate_output_signals(ax_so, n_freq, dtype):
        return csc_matrix(ax_so[np.newaxis, :].repeat(n_freq, axis=0), dtype=dtype)

    def reset_forward_signals(self):
        self.sax_si, self.sax_sn, self.sax_so, self.ax_sa, _ = \
            init_core_forward_signal(self.deep_network, self.dtype)

    def reset_backward_signals(self):
        self.sax_sib, self.sax_snb, self.sax_sob, _, _ = \
            init_core_backward_signal(self.deep_network, self.dtype)

    def reset_runner(self, reset_imputer=False):
        if reset_imputer:
            self.imputer.stream_features(is_cyclic=False)

        # reset signals
        self.reset_forward_signals()
        self.reset_backward_signals()

        return self

    def transform_array(self, max_iter=1000):

        ax_so, stop = None, False
        while not stop:
            # Get input signal
            sax_si = self.imputer.stream_next_forward()

            if sax_si is None or i > max_iter:
                stop = True
                continue

            # Transform and transmit forward
            self.sax_si = self.generate_input_signals(sax_si, self.deep_network.input_nodes, self.dtype)
            self.forward_transmit()

            if ax_so is None:
                ax_so = self.sax_so.toarray().sum(axis=0) > 0
            else:
                ax_so = np.vstack((ax_so, self.sax_so.toarray().sum(axis=0) > 0))

        return ax_so

    def reverse_array(self, ax_output, max_iter=10):
        raise NotImplementedError

    def forward_transmit(self):
        raise NotImplementedError

    def backward_transmit(self):
        raise NotImplementedError


def init_core_forward_signal(dn, dtype):

    N = dn.n_freq
    sax_si = csc_matrix((N, len(dn.input_nodes)), dtype=dtype)
    sax_sn = csc_matrix((N, len(dn.network_nodes)), dtype=dtype)
    sax_so = csc_matrix((N, len(dn.output_nodes)), dtype=dtype)
    ax_sa = np.array([False] * len(dn.network_nodes))
    sax_C = csc_matrix((len(dn.network_nodes), len(dn.output_nodes)))

    return sax_si, sax_sn, sax_so, ax_sa, sax_C


def init_core_backward_signal(dn, dtype):
    N = dn.n_freq

    sax_sib = csc_matrix((N, len(dn.input_nodes)), dtype=dtype)
    sax_snb = csc_matrix((N, len(dn.network_nodes)), dtype=dtype)
    sax_sob = csc_matrix((N, len(dn.output_nodes)), dtype=dtype)
    sax_sab = csc_matrix(np.array([[False] * len(dn.network_nodes)]).repeat(len(dn.output_nodes), axis=0))
    sax_Cb = csc_matrix((len(dn.network_nodes), len(dn.output_nodes)))

    return sax_sib, sax_snb, sax_sob, sax_sab, sax_Cb

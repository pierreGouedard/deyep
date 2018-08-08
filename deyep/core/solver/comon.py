# Global import
from scipy.sparse import csc_matrix
import numpy as np

# Local import
from deyep.core.tools.equations.fourrier import f_fnp, f_fcp, f_bop, f_bnp, f_bdu, f_bou, f_biu, f_buffer
from deyep.core.tools.equations.canonical import c_fnp, c_fcp, c_bop, c_bnp, c_bdu, c_bou, c_biu, c_buffer
from deyep.core.tools.equations.comon import fot, fnt, bnt, bit, bcv, bcu


class DeepNetSolver(object):
    def __init__(self, deep_network, delay, imputer, basis, p0):

        # Core params
        self.p = p0
        self.delay = delay
        self.basis = basis

        # Data structure
        self.deep_network = deep_network
        self.imputer = imputer
        self.l_keys_input = ['N={},k={}'.format(n.basis.N, n.basis.key) for n in self.deep_network.input_nodes]

        # Init signals
        self.dtype = int if basis == 'canonical' else complex
        self.sax_si, self.sax_sn, self.sax_so, self.ax_sa, self.sax_C = init_core_forward_signal(deep_network, self.dtype)
        self.sax_sib, self.sax_snb, self.sax_sob, self.sax_sab, self.sax_Cb = init_core_forward_signal(deep_network, self.dtype)

        self.t = 0
        self.t_fp = 0

    def run_epoch(self, n):
        print self.basis
        if self.basis == 'fourrier':
            self.run_epoch_fourrier(n)
        elif self.basis == 'canonical':
            self.run_epoch_canonical(n)
        else:
            raise ValueError('basis not recognized {}'.format(self.basis))

    def run_epoch_fourrier(self, n):

        i = 0

        while i < n:
            if self.t % 2 == 0:
                # Get new input and transmit forward
                self.sax_si = generate_input_signals(self.imputer.stream_next_forward(), self.deep_network.input_nodes,
                                                     self.dtype)
                self.forward_transmiting()

                # Run backward processing if delay is reached
                if self.t / 2 >= self.delay:
                    sax_got = self.imputer.stream_next_backward()
                    self.f_backward_processing(sax_got, self.t_fp - 1)

                i += 1

            else:
                # Run backward transmit
                if (self.t / 2) + 1 >= self.delay:
                    self.f_backward_transmiting(only_buffer=(self.t / 2) + 1 == self.delay)

                # Run forward processing
                self.f_forward_processing(self.t_fp)
                self.t_fp += 1
                i += 1

            self.t += 1

    def run_epoch_canonical(self, n):

        i = 0

        while i < n:
            if self.t % 2 == 0:
                # Get new input and transmit forward
                self.sax_si = generate_input_signals(self.imputer.stream_next_forward(), self.deep_network.input_nodes,
                                                     self.dtype)
                self.forward_transmiting()

                # Run backward processing if delay is reached
                if self.t / 2 >= self.delay:
                    sax_got = self.imputer.stream_next_backward()
                    self.c_backward_processing(sax_got, self.t_fp - 1)

                i += 1

            else:
                # Run backward transmit
                if (self.t / 2) + 1 >= self.delay:
                    self.c_backward_transmiting(only_buffer=(self.t / 2) + 1 == self.delay)

                # Run forward processing
                self.c_forward_processing(self.t_fp)
                self.t_fp += 1
                i += 1

            self.t += 1

    def run_fourrier_epoch_analysis(self, n):
        import time
        l_t_epoch, l_t_forwardp, l_t_forwardt, l_t_backwardp, l_t_backwardt, i = [], [], [], [], [], 0

        while i < n:
            t = time.time()
            if self.t % 2 == 0:
                # Get new input and transmit forward
                t0 = time.time()
                self.sax_si = generate_input_signals(self.imputer.stream_next_forward(), self.deep_network.input_nodes,
                                                     self.dtype)
                self.forward_transmiting()
                l_t_forwardt += [time.time() - t0]

                # Run backward processing if delay is reached
                if self.t / 2 >= self.delay:
                    t0 = time.time()
                    sax_got = self.imputer.stream_next_backward()
                    self.f_backward_processing(sax_got, self.t_fp - 1)
                    l_t_backwardp += [time.time() - t0]

                i += 1

            else:
                # Run backward transmit
                if (self.t / 2) + 1 >= self.delay:
                    t0 = time.time()
                    self.f_backward_transmiting(only_buffer=(self.t / 2) + 1 == self.delay)
                    l_t_backwardt += [time.time() - t0]

                # Run forward processing
                t0 = time.time()
                self.f_forward_processing(self.t_fp)
                l_t_forwardp += [time.time() - t0]
                self.t_fp += 1
                i += 1

            self.t += 1
            l_t_epoch += [time.time() - t]

        return l_t_epoch, l_t_forwardp, l_t_forwardt, l_t_backwardp, l_t_backwardt

    def run_canonical_epoch_analysis(self, n):
        import time
        l_t_epoch, l_t_forwardp, l_t_forwardt, l_t_backwardp, l_t_backwardt, i = [], [], [], [], [], 0

        while i < n:
            t = time.time()
            if self.t % 2 == 0:
                # Get new input and transmit forward
                t0 = time.time()
                self.sax_si = generate_input_signals(self.imputer.stream_next_forward(), self.deep_network.input_nodes,
                                                     self.dtype)
                self.forward_transmiting()
                l_t_forwardt += [time.time() - t0]

                # Run backward processing if delay is reached
                if self.t / 2 >= self.delay:
                    t0 = time.time()
                    sax_got = self.imputer.stream_next_backward()
                    self.c_backward_processing(sax_got, self.t_fp - 1)
                    l_t_backwardp += [time.time() - t0]

                i += 1

            else:
                # Run backward transmit
                if (self.t / 2) + 1 >= self.delay:
                    t0 = time.time()
                    self.c_backward_transmiting(only_buffer=(self.t / 2) + 1 == self.delay)
                    l_t_backwardt += [time.time() - t0]

                # Run forward processing
                t0 = time.time()
                self.c_forward_processing(self.t_fp)
                l_t_forwardp += [time.time() - t0]
                self.t_fp += 1
                i += 1

            self.t += 1
            l_t_epoch += [time.time() - t]

        return l_t_epoch, l_t_forwardp, l_t_forwardt, l_t_backwardp, l_t_backwardt

    def forward_transmiting(self):
        # Output transmit
        self.sax_so = fot(self.deep_network.O, self.sax_sn)

        # network transmit
        self.sax_sn = fnt(self.deep_network.D, self.deep_network.I, self.sax_sn, self.sax_si)

    def f_forward_processing(self, t):
        # transform signals
        self.sax_sn, self.ax_sa = f_fnp(self.sax_sn, self.deep_network.network_nodes, self.deep_network.tau, t)

        # Update candidate
        self.sax_C = f_fcp(self.ax_sa, self.deep_network.Cm)

    def f_backward_transmiting(self, only_buffer=False):

        if not only_buffer:
            # Cache result of backward transmit
            sax_snb_ = bnt(self.deep_network.D, self.deep_network.O, self.sax_snb, self.sax_sob, self.sax_sab)
            sax_sib_ = bit(self.deep_network.I, self.sax_snb)

            # Update deep network structure
            f_bdu(self.sax_snb, self.deep_network, penalty=self.p)
            f_bou(self.sax_sob, self.sax_sab, self.deep_network, penalty=self.p)
            f_biu(self.sax_snb, self.deep_network, penalty=self.p)
            bcu(self.sax_Cb, self.deep_network, w0=self.deep_network.w0)

            # Save result of backward transmit
            self.sax_snb = sax_snb_
            self.sax_sib = sax_sib_

        # Buffer forward signals
        self.sax_Cb, self.sax_sab, self.sax_sob = f_buffer(self.sax_C, self.ax_sa, len(self.deep_network.output_nodes),
                                                           self.sax_so)

    def f_backward_processing(self, sax_got, t):
        # Generate feedback
        l_nodes = [self.deep_network.network_nodes[i] for i in np.unique(self.sax_sab.nonzero()[1])]
        self.sax_sob = f_bop(self.sax_sob, sax_got, l_nodes)

        # Backward network process
        self.sax_snb = f_bnp(self.deep_network.network_nodes, self.sax_snb, t, self.l_keys_input)

        # Validate candidate
        self.sax_Cb = bcv(sax_got, self.sax_Cb)

    def c_forward_processing(self, t):
        # transform signals
        self.sax_sn, self.ax_sa = c_fnp(self.sax_sn, self.deep_network.network_nodes, self.deep_network.tau, t)

        # Update candidate
        self.sax_C = c_fcp(self.ax_sa, self.deep_network.Cm)

    def c_backward_transmiting(self, only_buffer=False):

        if not only_buffer:
            # Cache result of backward transmit
            sax_snb_ = bnt(self.deep_network.D, self.deep_network.O, self.sax_snb, self.sax_sob, self.sax_sab)
            sax_sib_ = bit(self.deep_network.I, self.sax_snb)

            # Update deep network structure
            c_bdu(self.sax_snb, self.deep_network, penalty=self.p)
            c_bou(self.sax_sob, self.sax_sab, self.deep_network, penalty=self.p)
            c_biu(self.sax_snb, self.deep_network, penalty=self.p)
            bcu(self.sax_Cb, self.deep_network, w0=self.deep_network.w0)

            # Save result of backward transmit
            self.sax_snb = sax_snb_
            self.sax_sib = sax_sib_

        # Buffer forward signals
        self.sax_Cb, self.sax_sab, self.sax_sob = c_buffer(self.sax_C, self.ax_sa, len(self.deep_network.output_nodes),
                                                           self.sax_so)

    def c_backward_processing(self, sax_got, t):

        # Generate feedback
        self.sax_sob = c_bop(self.sax_sob, sax_got)

        # Backward network process
        self.sax_snb = c_bnp(self.deep_network.network_nodes, self.sax_snb, t, self.l_keys_input)

        # Validate candidate
        self.sax_Cb = bcv(sax_got, self.sax_Cb)


def init_core_forward_signal(dn, dtype):

    N = dn.n_freq
    sax_si = csc_matrix((N, len(dn.input_nodes)), dtype=dtype)
    sax_sn = csc_matrix((N, len(dn.network_nodes)), dtype=dtype)
    sax_so = csc_matrix((N, len(dn.output_nodes)), dtype=dtype)
    ax_sa = np.array([False] * len(dn.network_nodes))
    sax_C = csc_matrix((len(dn.network_nodes), len(dn.network_nodes)))

    return sax_si, sax_sn, sax_so, ax_sa, sax_C


def init_core_backward_signal(dn):
    N = dn.n_freq

    sax_sib = csc_matrix((N, len(dn.input_nodes)), dtype=dtype)
    sax_snb = csc_matrix((N, len(dn.network_nodes)), dtype=dtype)
    sax_sob = csc_matrix((N, len(dn.output_nodes)), dtype=dtype)
    sax_sab = csc_matrix(np.array([[False] * len(dn.network_nodes)]).repeat(len(dn.output_nodes), axis=0))
    sax_Cb = csc_matrix((len(dn.network_nodes), len(dn.network_nodes)))

    return sax_sib, sax_snb, sax_sob, sax_sab, sax_Cb


def generate_input_signals(sax_i, input_nodes, dtype):
    sax_si = csc_matrix((len(input_nodes), input_nodes[0].basis.N), dtype=dtype)

    for i,  n in enumerate(input_nodes):
        if sax_i[0, i] >= 1:
            sax_si[i, :] = n.basis.base

    return sax_si.transpose()







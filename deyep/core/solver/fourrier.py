# Global import
from scipy.sparse import csc_matrix
import numpy as np

# Local import
from deyep.core.tools.equations.fourrier import f_fnp, f_fcp, f_bop, f_bnp, f_bdu, f_bou, f_biu, f_buffer
from deyep.core.tools.equations.comon import fot, fnt, bnt, bit, bcv, bcu
from deyep.core.solver.comon import DeepNetSolver


class FourrierDeepNetSolver(DeepNetSolver):
    def __init__(self, deep_network, delay, imputer, p0, verbose=0):
        DeepNetSolver.__init__(self, deep_network, delay, imputer, p0, 'fourrier', verbose=verbose)

    def run_epoch(self, n):

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
                    self.backward_processing(sax_got, self.t_fp - 1)

                i += 1

            else:
                # Run backward transmit
                if (self.t / 2) + 1 >= self.delay:
                    self.backward_transmiting(only_buffer=(self.t / 2) + 1 == self.delay)

                # Run forward processing
                self.forward_processing(self.t_fp)
                self.t_fp += 1
                i += 1

            if self.verbose == 1:
                print self.t
            self.t += 1

    def epoch_analysis(self, n):
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
                    self.backward_processing(sax_got, self.t_fp - 1)
                    l_t_backwardp += [time.time() - t0]

                i += 1

            else:
                # Run backward transmit
                if (self.t / 2) + 1 >= self.delay:
                    t0 = time.time()
                    self.backward_transmiting(only_buffer=(self.t / 2) + 1 == self.delay)
                    l_t_backwardt += [time.time() - t0]

                # Run forward processing
                t0 = time.time()
                self.forward_processing(self.t_fp)
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

    def forward_processing(self, t):
        # transform signals
        self.sax_sn, self.ax_sa = f_fnp(self.sax_sn, self.deep_network.network_nodes, self.deep_network.tau, t)

        # Update candidate
        self.sax_C = f_fcp(self.ax_sa, self.deep_network.Cm, self.sax_C)

    def backward_transmiting(self, only_buffer=False):

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

    def backward_processing(self, sax_got, t):
        # Generate feedback
        l_nodes = [self.deep_network.network_nodes[i] for i in np.unique(self.sax_sab.nonzero()[1])]
        self.sax_sob = f_bop(self.sax_sob, sax_got, l_nodes)

        # Backward network process
        self.sax_snb = f_bnp(self.deep_network.network_nodes, self.sax_snb, t, self.key_inputs)

        # Validate candidate
        self.sax_Cb = bcv(sax_got, self.sax_Cb)

# Global import
from scipy.sparse import csc_matrix
import numpy as np

# Local import
from deyep.core.tools.equations.backward import bop, bnp, bnt, bit, buffer
from deyep.core.tools.equations.forward import fnp, fot, fnt
from deyep.core.tools.equations.structure import bdu, bou, biu


class FiringGraphDrainer(object):
    def __init__(self, p, firing_graph, imputer, depth=2, verbose=0):

        # Core params
        self.p = p
        self.firing_graph = firing_graph
        self.depth = depth
        self.verbose = verbose

        # D
        self.imputer = imputer
        self.key_inputs = set(['k={},N={}'.format(n.basis.key, n.basis.N) for n in self.firing_graph.input_vertices])

        # Init signals
        self.sax_si, self.sax_sn, self.sax_so, self.ax_sa = \
            FiringGraphDrainer.init_core_forward_signal(self.firing_graph)
        self.sax_sib, self.sax_snb, self.sax_sob, self.sax_sab = \
            FiringGraphDrainer.init_core_backward_signal(self.firing_graph)

        self.tmp_analysis = 0
        self.tmp_tracked_input_vertex = 1
        self.t = 0
        self.t_fp = 0

    def reset_all(self, reset_imputer=False, reset_time=False, reset_inputs=False):
        if reset_imputer:
            self.imputer.stream_features()

        if reset_time:
            self.t, self.t_fp = 0, 0

        if reset_inputs:
            self.key_inputs = set(['k={},N={}'.format(n.basis.key, n.basis.N) for n in self.firing_graph.input_vertices])

        # reset signals
        self.reset_forward_signals()
        self.reset_backward_signals()

    def reset_forward_signals(self):
        self.sax_si, self.sax_sn, self.sax_so, self.ax_sa = \
            FiringGraphDrainer.init_core_forward_signal(self.firing_graph)

    def reset_backward_signals(self):
        self.sax_sib, self.sax_snb, self.sax_sob, self.sax_sab = \
            FiringGraphDrainer.init_core_backward_signal(self.firing_graph)

    def drain(self, n):

        i = 0
        while i < n:
            if self.t % 2 == 0:
                # Get new input and transmit forward
                self.sax_si = self.generate_input_signals(
                    self.imputer.stream_next_forward(), self.firing_graph.input_vertices
                )
                self.forward_transmiting()

                # Run backward processing if delay is reached
                if self.t / 2 >= self.depth:
                    sax_got = self.imputer.stream_next_backward()
                    self.backward_processing(sax_got, self.t_fp - 1)

                i += 1

            else:
                # Run backward transmit
                if (self.t / 2) + 1 >= self.depth:
                    self.backward_transmiting(only_buffer=(self.t / 2) + 1 == self.depth)

                # Run forward processing
                self.forward_processing(self.t_fp)

                self.t_fp += 1
                i += 1

            self.t += 1

            if self.verbose == 1 and self.t % 100 == 0:
                print '[Drainer Info]: iteration {} completed'.format(self.t, n)

        return self

    def drain_analyse(self, n):
        import time
        l_t_epoch, l_t_forwardp, l_t_forwardt, l_t_backwardp, l_t_backwardt, i = [], [], [], [], [], 0

        while i < n:
            t = time.time()
            if self.t % 2 == 0:
                # Get new input and transmit forward
                t0 = time.time()
                self.sax_si = self.generate_input_signals(
                    self.imputer.stream_next_forward(), self.firing_graph.input_vertices
                )
                self.forward_transmiting()
                l_t_forwardt += [time.time() - t0]

                # Run backward processing if delay is reached
                if self.t / 2 >= self.depth:
                    t0 = time.time()
                    sax_got = self.imputer.stream_next_backward()
                    self.backward_processing(sax_got, self.t_fp - 1)
                    l_t_backwardp += [time.time() - t0]

                i += 1

            else:
                # Run backward transmit
                if (self.t / 2) + 1 >= self.depth:
                    t0 = time.time()
                    self.backward_transmiting(only_buffer=(self.t / 2) + 1 == self.depth)
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
        self.sax_so = fot(self.firing_graph.O, self.sax_sn)

        # network transmit
        self.sax_sn = fnt(self.firing_graph.D, self.firing_graph.I, self.sax_sn, self.sax_si)

        # Udpate forward tracking
        self.firing_graph.update_forward_firing(self.sax_si, self.sax_sn, self.sax_so)

    def forward_processing(self, t):
        # transform signals
        self.sax_sn, self.ax_sa = fnp(self.sax_sn, self.firing_graph.core_vertices, t)

    def backward_transmiting(self, only_buffer=False):

        if not only_buffer:
            # Cache result of backward transmit
            sax_snb_ = bnt(self.firing_graph.D, self.firing_graph.O, self.sax_snb, self.sax_sob, self.sax_sab)
            sax_sib_ = bit(self.firing_graph.I, self.sax_snb)

            # Update firing graph structure
            sax_Du, sax_Ou, sax_Iu = None, None, None
            if self.firing_graph.update_policy('D'):
                sax_Du = bdu(self.sax_snb, self.firing_graph, penalty=self.p)
                sax_Ou = bou(self.sax_sob, self.sax_sab, self.firing_graph, penalty=self.p)

            if self.firing_graph.update_policy('I'):
                sax_Iu = biu(self.sax_snb, self.firing_graph, penalty=self.p)

            self.firing_graph.update_bakward_firing(sax_Iu, sax_Du, sax_Ou)

            # Save result of backward transmit
            self.sax_snb = sax_snb_
            self.sax_sib = sax_sib_

        # Buffer forward signals
        self.sax_sab, self.sax_sob = buffer(self.ax_sa, len(self.firing_graph.output_vertices), self.sax_so)

    def backward_processing(self, sax_got, t):

        # Generate feedback
        self.sax_sob = bop(self.sax_sob, sax_got)

        # Backward network process
        self.sax_snb = bnp(self.firing_graph.core_vertices, self.sax_snb, t, self.key_inputs)


    @staticmethod
    def generate_input_signals(sax_i, input_vertices):
        sax_si = csc_matrix((len(input_vertices), input_vertices[0].basis.N), dtype=int)

        for i,  n in enumerate(input_vertices):
            if sax_i[0, i] >= 1:
                sax_si[i, :] = n.basis.base

        return sax_si.transpose()

    @staticmethod
    def init_core_forward_signal(fg):

        N = fg.n_freq
        sax_si = csc_matrix((N, len(fg.input_vertices)), dtype=int)
        sax_sn = csc_matrix((N, len(fg.core_vertices)), dtype=int)
        sax_so = csc_matrix((N, len(fg.output_vertices)), dtype=int)
        ax_sa = np.array([False] * len(fg.core_vertices))

        return sax_si, sax_sn, sax_so, ax_sa

    @staticmethod
    def init_core_backward_signal(fg):
        N = fg.n_freq

        sax_sib = csc_matrix((N, len(fg.input_vertices)), dtype=int)
        sax_snb = csc_matrix((N, len(fg.core_vertices)), dtype=int)
        sax_sob = csc_matrix((N, len(fg.output_vertices)), dtype=int)
        sax_sab = csc_matrix(np.array([[False] * len(fg.core_vertices)]).repeat(len(fg.output_vertices), axis=0))

        return sax_sib, sax_snb, sax_sob, sax_sab

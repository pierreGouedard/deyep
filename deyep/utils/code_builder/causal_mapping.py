# Global import
import numpy as np
from scipy.sparse import csc_matrix

# Local import
from deyep.utils.code_builder.comon import CodeBuilder
from deyep.utils.code_builder.simple_mapping import SimpleMapping


class CausalMapping(CodeBuilder):

    def __init__(self, size, length, l_spread, p=0.5, n_sequence=1000, sparse=True):
        CodeBuilder.__init__(self, size, n_sequence=n_sequence, sparse=sparse)
        self.spread = l_spread
        self.length = map(int, length)
        self.simple_maping = SimpleMapping(self.size, self.length, p, self.n_sequence, sparse=False)

    def generate_code(self):
        self.simple_maping.generate_code()
        self.code, self.rcode = {}, {}

        for _code, code_ in self.simple_maping.code.items():

            # Create causal random sequence
            c_code_in = np.random.binomial(1, 0.5, (np.random.randint(1, self.spread), self.length[0]))
            c_code_out = np.random.binomial(1, 0.5, (np.random.randint(1, self.spread), self.length[1]))

            # equalize spread
            t_max = max(c_code_in.shape[0], c_code_out.shape[0])

            if c_code_in.shape[0] < t_max:
                c_code_in = np.vstack((c_code_in, np.zeros((t_max - c_code_in.shape[0], self.length[0]))))
            elif c_code_out.shape[0] < t_max:
                c_code_out = np.vstack((c_code_out, np.zeros((t_max - c_code_out.shape[0], self.length[0]))))

            # Store causal code in dict
            self.code.update({_code: map(bool, c_code_in)})
            self.rcode.update({code_: map(bool, c_code_out)})

        return self

    def encode(self, ax_code):
        ax_code_ = self.simple_maping.encode(ax_code)
        CodeBuilder.encode(self, ax_code_)

    def decode(self, ax_code):
        ax_code_ = self.simple_maping.decode(ax_code)
        CodeBuilder.decode(self, ax_code_)
            
    def generate_random_sequence(self, offset=0):

        ax_input, ax_output = np.zeros((1, self.length[0])), np.zeros((1, self.length[1]))

        ax_input_, ax_output_ = self.simple_maping.generate_random_sequence(offset=offset)

        for i, ax_in in enumerate(ax_input_):
            ax_input = np.vstack((ax_input, self.encode(ax_in)))
            if ax_output_[i, :].sum() > 0:
                ax_output = np.vstack((ax_input, self.decode(ax_output_[i, :])))

        if self.sparse:
            return csc_matrix(ax_input[1:, :]), csc_matrix(ax_output[1:, :])

        return ax_input[1:, :], ax_output[1:, :]


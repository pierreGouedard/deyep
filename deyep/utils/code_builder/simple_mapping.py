# Global import
import numpy as np
import random
from scipy.sparse import csc_matrix

# Local import
from deyep.utils.code_builder.comon import CodeBuilder


class SimpleMapping(CodeBuilder):

    def __init__(self, size, length, p=0.5, n_sequence=1000, sparse=True):
        CodeBuilder.__init__(self, size, n_sequence=n_sequence, sparse=sparse)
        self.length = map(int, length)
        self.p = p

    def generate_code(self):
        self.code, self.rcode = {}, {}

        for i in range(self.size):
            # Create two random code
            code = tuple(map(bool, np.random.binomial(1, self.p, self.length[0])))
            code_ = tuple(map(bool, np.random.binomial(1, self.p, self.length[1])))

            # Map them in code
            self.code.update({code: code_})
            self.rcode.update({code_: code})

        return self

    def generate_random_sequence(self, offset=0):
        l_codes = self.code.keys()
        ax_input, ax_output = np.zeros([1, self.length[0]]), np.zeros([1 + offset, self.length[1]])

        for i in range(self.n_sequence):
            code = random.choice(l_codes)
            ax_input = np.vstack((ax_input, np.array(code)))
            ax_output = np.vstack((ax_output, np.array(self.code[code])))

        if self.sparse:
            return csc_matrix(ax_input[1:, :]), csc_matrix(ax_output[1:, :])

        return ax_input[1:, :], ax_output[1:, :]


# Global import
import numpy as np
import random
from scipy.sparse import csc_matrix


class CodeBuilder(object):

    def __init__(self, size, length, p=0.5, seed=1234, n_sequence=1000, sparse=True):
        self.size = size
        self.length = map(int, length)
        self.p = p
        self.code = {}
        self.rcode = {}
        self.seed = seed
        self.n_sequence = n_sequence
        self.sparse = sparse

    def generate_code(self):

        for i in range(self.size):
            # Create two random code
            code = tuple(map(bool, np.random.binomial(1, self.p, self.length[0])))
            code_ = tuple(map(bool, np.random.binomial(1, self.p, self.length[1])))

            # Map them in code
            self.code.update({code: code_})
            self.rcode.update({code_: code})

        return self

    def encode(self, ax_code):
        code_ = self.code[tuple(map(bool, ax_code))]
        return np.array(code_)

    def decode(self, ax_code):
        code = self.rcode[tuple(map(bool, ax_code))]
        return np.array(code)

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

    def save(self, path, name_input, name_output, driver, offset=0):
        sax_input, sax_output = self.generate_random_sequence(offset=offset)

        # Create I/O and save it into tmpdir files
        driver.write_file(sax_input, driver.join(path, name_input), is_sparse=True)
        driver.write_file(sax_output, driver.join(path, name_output), is_sparse=True)

        return self









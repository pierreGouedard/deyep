# Global import
import numpy as np
import random
from scipy.sparse import csc_matrix

# Local import
from deyep.utils.driver.nmp import NumpyDriver


class CodeBuilder(object):
    driver = NumpyDriver()

    def __init__(self, size, n_sequence=1000, sparse=True):
        self.size = size
        self.code = None
        self.rcode = None
        self.n_sequence = n_sequence
        self.sparse = sparse

    def generate_code(self):
        raise NotImplementedError

    def encode(self, ax_code):
        if self.code is None:
            self.generate_code()

        code_ = self.code[tuple(map(bool, ax_code))]
        return np.array(code_)

    def decode(self, ax_code):
        if self.code is None:
            self.generate_code()

        code = self.rcode[tuple(map(bool, ax_code))]
        return np.array(code)

    def generate_random_sequence(self, offset=0):
        raise NotImplementedError

    def save(self, path, name_input, name_output, offset=0):
        sax_input, sax_output = self.generate_random_sequence(offset=offset)

        # Create I/O and save it into tmpdir files
        self.driver.write_file(sax_input, self.driver.join(path, name_input), is_sparse=self.sparse)
        self.driver.write_file(sax_output, self.driver.join(path, name_output), is_sparse=self.sparse)

        return self

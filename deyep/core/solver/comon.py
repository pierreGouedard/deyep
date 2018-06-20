

class DeepNetSolver(object):

    def __init__(self, deep_network, delay, imputer):

        self.deep_network = deep_network
        self.imputer = imputer
        self.delay = delay

    def run_epoch(self, n):
        raise NotImplementedError

# Global import

# Local import
from deyep.core.builder.binomial import BinomialGraphBuilder
from deyep.utils.code_builder import CodeBuilder
from deyep.core.imputer import identity
from deyep.utils.simulations import Simulation


class CodeBreaker(Simulation):

    name = 'code_breaker'
    params_network = {'ni': 10, 'nd': 100, 'no': 5, 'depth': 2, 'p0': 0.1, 'l0': 10, 'tau': 5, 'w0': 10,
                      'basis': "canonical", 'capacity': 10, 'delay': 0}
    imputer = identity.DoubleIdentityImputer
    builder = BinomialGraphBuilder
    code_builder = CodeBuilder(20, [n_i, n_o], p=0.5, seed=1234).generate_code()

    def __init__(self, resume=False):
        Simulation.__init__(self, CodeBreaker.name, CodeBreaker.imputer, CodeBreaker.params_network,
                            CodeBreaker.builder, code_builder=CodeBreaker.code_builder, resume=resume)

    def check_network_cleaning(self):
        raise NotImplementedError


import IPython
IPython.embed()
# Interesting metrics:
#   * Optimality score (is this a sub network of optimal ?, how many time a penalty is sent ?)
#   * completness score (how far from complete optimal we are)
#   * efficiency score (does the compression is  correct) how far from optimal compression ? complexity


# TODO:
# Merge multiple networks


if __name__ == '__main__':
    print 'Put the code of simulation right here motherfucker'

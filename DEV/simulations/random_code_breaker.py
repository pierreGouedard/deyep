# Global import
import sys

# Local import
from deyep.core.builder.binomial import BinomialGraphBuilder
from deyep.utils.code_builder.simple_mapping import SimpleMapping
from deyep.core.imputer import array
from deyep.utils.simulations import Simulation


class CodeBreaker(Simulation):
    name = 'code_breaker'
    params_network = {'ni': 10, 'nd': 100, 'no': 5, 'depth': 3, 'p0': 0.1, 'l0': 10, 'tau': 5, 'w0': 10, 'capacity': 5}
    imputer = array.DoubleArrayImputer
    builder = BinomialGraphBuilder
    raw_builder = SimpleMapping(20, [10, 5], p=0.5).generate_code()

    def __init__(self, resume=False):
        Simulation.__init__(
            self, CodeBreaker.name, CodeBreaker.imputer, CodeBreaker.params_network, CodeBreaker.builder,
            raw_builder=CodeBreaker.raw_builder, resume=resume
        )


if __name__ == '__main__':
    #TODO: Before launching run: export PYTHONPATH="/home/erepie/deyep/" => fix it
    l_args = []

    if len(sys.argv) > 1:
        l_args = sys.argv[1:]

    if 'resume' in l_args:
        sim = CodeBreaker(resume=True)
    else:
        sim = CodeBreaker()

    sim.fit_network(verbose=2)
    import IPython
    IPython.embed()

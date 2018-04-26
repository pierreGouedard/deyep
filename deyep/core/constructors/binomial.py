# Global import

# Local import
from deyep.core.constructors.constructors import Constructor


class BinomialConstructor(Constructor):

    def __init__(self, project, seed, feature_size, edge_density, w0):
        # Init inherited class attribute
        Constructor.__init__(self, project, seed, feature_size, edge_density, w0)

        # Init specific attributes


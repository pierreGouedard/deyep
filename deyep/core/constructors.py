# Global imports

# local import
import settings

class Constructors(object):

    def __init__(self, project, seed, feature_size, edge_density, edge_weight):

        # settings
        self.project = project
        self.seed = seed
        self.url_in = settings.deyep_io_path.format(project)
        self.url_out = os.path.join(settings.deyep_network_path.format(project), 'init_{}.json'.format(seed))

        # Params
        self.feature_size = feature_size
        self.edge_density = edge_density
        self.edge_weight = edge_weight

        # Nodes
        self.input_nodes = []
        self.output_nodes = []
        self.network_nodes = []

        # Edges
        self.input_edges = []
        self.output_edges = []
        self.network_edges = []

    def build_input_nodes(self):
        raise NotImplementedError

    def build_output_nodes(self):
        raise NotImplementedError

    def build_network_nodes(self):
        raise NotImplemented

    def build_edges_input(self):
        raise NotImplementedError

    def build_network_edges(self):
        raise NotImplementedError

    def save_network(self):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError
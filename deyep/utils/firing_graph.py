# Global imports
import numpy as np
from scipy.sparse import eye

# Local import
from firing_graph.graph import FiringGraph, create_empty_matrices
from deyep.models.firing_graph_models import FgComponents
from deyep.linalg.spmat_op import fill_gap


class DeyepFiringGraph(FiringGraph):
    """
    This class implement the main data structure used for fitting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """

    def __init__(self, **kwargs):
        kwargs.update({'project': 'DeyepFiringGraph', 'depth': 2})

        # Invoke parent constructor
        super(DeyepFiringGraph, self).__init__(**kwargs)

    def to_comp(self):
        return FgComponents(inputs=self.I, levels=self.levels, meta=self.meta)

    @staticmethod
    def from_comp(fg_comp: FgComponents):

        if len(fg_comp) == 0:
            return None

        # Get output indices and initialize matrices
        d_matrices = create_empty_matrices(
            fg_comp.inputs.shape[0], fg_comp.inputs.shape[1], fg_comp.inputs.shape[1], write_mode=False
        )

        # Set matrices
        d_matrices['Iw'] = fg_comp.inputs.astype(np.int32)
        d_matrices['O'] += eye(fg_comp.inputs.shape[1], format='csr', dtype=bool)

        # Add firing graph kwargs
        kwargs = {
            'meta': fg_comp.meta, 'matrices': d_matrices, 'ax_levels': fg_comp.levels,
        }

        return DeyepFiringGraph(**kwargs)

    def get_convex_hull(self, server):
        # Get masked activations
        sax_x = server.next_all_forward().sax_data_forward

        # propagate through firing graph
        sax_fg = self.seq_propagate(sax_x)

        # Get masked activations
        sax_product = sax_x.T.dot(sax_fg)

        return FgComponents(
            inputs=fill_gap(sax_product, server.bitmap), levels=np.ones(sax_fg.shape[1]), meta=self.meta
        )


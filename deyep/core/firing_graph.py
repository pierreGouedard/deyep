# Global imports
import numpy as np
from itertools import groupby
from scipy.sparse import eye, csr_matrix

# Local import
from firing_graph.graph import FiringGraph
from firing_graph.graph import create_empty_matrices
from deyep.core.data_models import FgComponents
from deyep.core.spmat_op import fill_gap


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
        return FgComponents(inputs=self.I, mask_inputs=self.Im, levels=self.levels, partitions=self.partitions)

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
        d_matrices['Im'] = fg_comp.mask_inputs
        d_matrices['O'] += eye(fg_comp.inputs.shape[1], format='csr', dtype=bool)

        # Add firing graph kwargs
        kwargs = {
            'partitions': fg_comp.partitions, 'matrices': d_matrices, 'ax_levels': fg_comp.levels,
        }

        return DeyepFiringGraph(**kwargs)

    @staticmethod
    def from_inputs(sax_inputs, levels, partitions, sax_mask_inputs=None):

        # Get output indices and initialize matrices
        d_matrices = create_empty_matrices(
            sax_inputs.shape[0], sax_inputs.shape[1], sax_inputs.shape[1], write_mode=False
        )

        # Set matrices
        d_matrices['Iw'] = sax_inputs.astype(np.int32)
        d_matrices['Im'] = sax_mask_inputs
        d_matrices['O'] += eye(sax_mask_inputs.shape[1], format='csr', dtype=bool)

        # Add firing graph kwargs
        kwargs = {
            'partitions': partitions, 'matrices': d_matrices, 'ax_levels': levels,
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
            inputs=fill_gap(sax_product, server.bitmap), mask_inputs=csr_matrix((0, 0)),
            levels=np.ones(sax_fg.shape[1]), partitions=self.partitions
        )


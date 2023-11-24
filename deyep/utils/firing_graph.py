# Global imports
import numpy as np
from scipy.sparse import eye

# Local import
from firing_graph.graph import FiringGraph, create_empty_matrices
from deyep.models.base import FgComponents
from deyep.linalg.spmat_op import fill_gap


class DeyepFiringGraph(FiringGraph):
    """
    DeyepFiringGraph inherit from the base firing graph. It correspond to a graph with 3 main matrix.

    the firing graph is composed of 3 kind of vertices:
     * input vertices: First layer of vertex, directly link to a potential input signal.
     * core vertices: Internal vertices, receive and send signal only from, to vertices of the firing graph.
     * output vertices: Last layer of vertices that receive signal from core vertices and don't send signals.

    Each vertex has a level, which correspond to the minimum number of positive signal required for the vertex signal
    to take value 1, the vertex value takes value 0 otherwise.

    The firing graph provide a procedure to iteratively propagate an input signal composed of sparse signal
    taking values 0, 1 through its vertices. There is 3 type of matrices that enable the propagation.
    Let ni, nc and no be respectively the number of input vertices

     * input matrix: enable to propagate the input signal ({0, 1} values) to the first layer of core vertices.
     * core matrix: enable to propagate signal from core vertex to core vertex propagation of signal
     * outpur matrix: enable to propagate the vertices signal to output vertices

    DeyepFiringGraph is a particular form of firing graph of depth 2 with empty core matrix. In addition the first
    layer of vertices is directly link to output vertices using a 1-1 unique mapping (ni == no).

    """

    def __init__(self, **kwargs):
        kwargs.update({'project': 'DeyepFiringGraph', 'depth': 2})

        # Invoke parent constructor
        super(DeyepFiringGraph, self).__init__(**kwargs)

    def to_comp(self):
        return FgComponents(inputs=self.I, levels=self.levels, _meta=self.meta)

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
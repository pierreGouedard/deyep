# Global import
import numpy as np

# Local import
from deyep.core.firing_graph import DeyepFiringGraph
from deyep.core.spmat_op import expand, explode
from deyep.core.data_models import BitMap, FgComponents


class ShapeEncoder:
    def __init__(self, bitmap):
        self.bitmap = bitmap

    def encode_shape(self, sax_i, components):
        # Explode shape
        sax_exploded_map = explode(components.inputs, self.bitmap, components.partitions)
        sax_exploded_mask = explode(components.inputs, self.bitmap, components.partitions, use_bf_mask=True)

        # Build up and dow
        sax_exploded_up = sax_exploded_mask + expand(
            sax_exploded_map, self.bitmap, keep_only_expanded=True, is_only_up=True
        )
        sax_exploded_down = sax_exploded_mask + expand(
            sax_exploded_map, self.bitmap, keep_only_expanded=True, is_only_down=True
        )

        # Propagate signal
        sax_x_up = DeyepFiringGraph.from_comp(FgComponents(sax_exploded_up).autofill_missing(self.bitmap.nf))\
            .seq_propagate(sax_i)
        sax_x_down = DeyepFiringGraph.from_comp(FgComponents(sax_exploded_down).autofill_missing(self.bitmap.nf))\
            .seq_propagate(sax_i)

        # reshape N to Height x width
        ax_code = np.hstack([
            sax_x_up.sum(axis=0).A.reshape((len(components), self.bitmap.nf)),
            sax_x_down.sum(axis=0).A.reshape((len(components), self.bitmap.nf)),
        ])

        return ax_code



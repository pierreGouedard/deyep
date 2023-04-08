# Global import
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
from scipy.sparse import spmatrix, hstack
from itertools import groupby
from operator import itemgetter

# Local import
from deyep.utils.firing_graph import DeyepFiringGraph
from deyep.linalg.spmat_op import expand, explode
from deyep.utils.discretizer import ImgCoordDiscretizer
from deyep.models.firing_graph_models import FgComponents, BitMap
from deyep.shapes.drawer import ShapeDrawer
from deyep.utils.plots import bar_plot
from deyep.models.img_graph_models import Node, Chain


class ImageEncoder:
    # TODO Last piece to add is the open cv segmenter to generate masks
    def __init__(self, l_masks: List[np.array], n_features: int):
        self.pixel_shape = l_masks[0].shape[:2]
        self.masks = l_masks

        # Discretize images
        self.data, self.flag, self.bitmap, self.img_discretizer = self.discretize_image(n_features)

    def discretize_image(self, n_features):
        # Encode pixels coordinates
        img_discretizer = ImgCoordDiscretizer(max(self.pixel_shape), n_features, 1, 0.0).fit(self.masks[0])
        data = img_discretizer.transform(img_discretizer.augmented_basis)

        # Encode labels
        flag = hstack([
            img_discretizer.transform_labels(ax_img[img_discretizer.basis[:, 0], img_discretizer.basis[:, 1]])
            for ax_img in self.masks
        ])

        # get bitmap
        bitmap = BitMap(img_discretizer.bf_map, img_discretizer.basis, img_discretizer.transformation)

        return data, flag, bitmap, img_discretizer

    def encode_image(self) -> Chain:
        # Build chain tree for each masks
        chain_trees = self.build_chain_tree()

        # Reduce Chain Trees
        l_chains = self.reduce_chain_trees(chain_trees)

        # Merge all chains
        chain = self.merge_chains()

        return chain

    def build_chain_tree(self):
        # TODO: This part of the code is dedicated to build the tree that will be used for mask chain encoding
        #
        # Here we should have a depth first search strategy, stop searching when convex defects is too small
        # This part of the code invoke MaskEncoder
        # As a consequence Mask Encoder does not need to do the image discretization.
        encode_masks(self.data, self.flag, self.bitmap, self.pixel_shape)
        import IPython
        IPython.embed()

        # TODO:
        #  1. compute convex hull + get chain and define as the root of shape tree.
        #  2. find children of CH => CH . !flag that are connexe (Here Opencv should be used)
        #  3. for each child > get convex hull + get chain and set the child in tree
        #  4. iterate over child, until no more child to run through
        #   => can we do that in parallel for multiple shapes? => Try

        pass

    def reduce_chain_trees(self):
        # TODO: merge chains that are part of the same shape (vonvex defect tree
        pass

    def merge_image_chain(self):
        # TODO: merge chains that represent shapes
        pass


def encode_masks(data: spmatrix, flag: spmatrix, bitmap: BitMap, pixel_shape):
    # TODO => the drawer draw it 'penché' needs to investigate that before getting to next step (Thursday)
    #   So that, this weekend  build_chain_tree is coded & end of next week reduce_chain_trees
    #   After that there will be some theory needed.
    # Build input firing graph
    sax_i = data.T.dot(flag)

    # Encode shape
    d_chains = build_chains(data, FgComponents(sax_i).set_levels(bitmap.nf), bitmap)

    # Show non sparse scales
    bar_plot('Encoding as bar plot', d_chains[0].scales(sparse=False, n_dir=bitmap.nf * 2))

    # Draw shape from encoding (tmp for dev purpose)
    shape_drawer = ShapeDrawer(pixel_shape)
    shape_drawer.draw_shape(d_chains[0])


def build_chains(sax_data, comp: FgComponents, bitmap: BitMap) -> Dict[int, Chain]:
    # Explode shape
    sax_exploded_map = explode(comp.inputs, bitmap, comp.meta)
    sax_exploded_mask = explode(comp.inputs, bitmap, comp.meta, use_bf_mask=True)

    # Build up and dow
    sax_exploded_up = sax_exploded_mask + expand(
        sax_exploded_map, bitmap, keep_only_expanded=True, is_only_up=True
    )
    sax_exploded_down = sax_exploded_mask + expand(
        sax_exploded_map, bitmap, keep_only_expanded=True, is_only_down=True
    )

    # Propagate signal
    sax_x_up = DeyepFiringGraph.from_comp(FgComponents(sax_exploded_up).set_levels(bitmap.nf))\
        .seq_propagate(sax_data)
    sax_x_down = DeyepFiringGraph.from_comp(FgComponents(sax_exploded_down).set_levels(bitmap.nf))\
        .seq_propagate(sax_data)

    # Get nodes for each comp
    d_nodes = build_nodes(bitmap, sax_x_up, len(comp))
    d_nodes = build_nodes(bitmap, sax_x_down, len(comp), d_nodes=d_nodes, down=True)

    # Build chains
    return {k: Chain(v) for k, v in d_nodes.items()}


def build_nodes(
        bitmap: BitMap, sax_x: spmatrix, n_comp: int, d_nodes: Dict[int, List[Node]] = None,
        down: bool = False
) -> Dict[int, List[Node]]:

    # Precompute scales and offset coef
    ax_scales = sax_x.sum(axis=0).A[0]
    offset_coef = int(np.pi) / bitmap.nf

    # Init list of nodes if necessary
    if d_nodes is None:
        d_nodes = {i: [] for i in range(n_comp)}
    # Build nodes for each comp
    for cind, l_sub_inds in groupby(sorted(zip(*sax_x.nonzero()), key=itemgetter(1)), itemgetter(1)):
        l_linds, norm_ind = list(map(itemgetter(0), l_sub_inds)), int((cind % bitmap.nf) + bitmap.nf / 2)
        if down:
            # Get min position
            arg_min = (bitmap.projections[l_linds, norm_ind % bitmap.nf] * ((2 * (norm_ind >= bitmap.nf)) - 1)).argmin()

            # Apply offset in the inverse direction of normal of current direction
            offset_dir = (((2 * (norm_ind < bitmap.nf)) - 1) * bitmap.transform_op[:, norm_ind % bitmap.nf])
            p = bitmap.basis[l_linds][arg_min] + offset_coef * offset_dir
        else:
            arg_min = (bitmap.projections[l_linds, norm_ind % bitmap.nf] * ((2 * (norm_ind < bitmap.nf)) - 1)).argmin()

            # Apply offset in the inverse direction of normal of current direction
            offset_dir = (((2 * (norm_ind >= bitmap.nf)) - 1) * 6 * bitmap.transform_op[:, norm_ind % bitmap.nf])
            p = bitmap.basis[l_linds][arg_min] + offset_coef * offset_dir

        # Set Node
        d_nodes[cind // bitmap.nf].append(Node(
            direction=(cind % bitmap.nf) + bitmap.nf * down, p0=tuple(p.astype(int)), scale=ax_scales[cind]
        ))

    return d_nodes

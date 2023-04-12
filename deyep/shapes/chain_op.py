# Global import
from typing import List, Dict, Tuple
import numpy as np
from scipy.sparse import spmatrix, hstack, csc_matrix
from itertools import groupby
from operator import itemgetter
from treelib import Node, Tree

# Local import
from deyep.utils.firing_graph import DeyepFiringGraph
from deyep.linalg.spmat_op import expand, explode
from deyep.models.base import FgComponents, BitMap
from deyep.shapes.drawer import ShapeDrawer
from deyep.utils.plots import bar_plot
from deyep.models.image import Node, SimpleChain
from deyep.opencv.utils import get_mask_cc


def merge_simple_chain(l_simple_chain: List[SimpleChain], bitmap: BitMap):
    pass


def reduce_simple_chain_trees(d_chain_tree: Dict[str, Tree], bitmap: BitMap):
    pass


def build_simple_chain_trees(
        sax_flags: spmatrix, sax_data: spmatrix, bitmap: BitMap, pixel_shape: Tuple[int, int]
) -> Dict[str, Tree]:
    # At start, flags are init flags
    l_meta = [{'id': '::' + FgComponents.rd_id(5)} for _ in range(sax_flags.shape[1])]
    d_trees = {d['id'].split(':')[-1]: Tree() for d in l_meta}

    # Until there is no more flag
    while sax_flags.nnz > 0:

        # Create FgComp of flag's convex hull
        fg_comps = FgComponents(sax_data.T.dot(sax_flags)).set_levels(bitmap.nf).set_meta(l_meta)
        sax_ch_data = DeyepFiringGraph.from_comp(fg_comps).seq_propagate(sax_data)

        # encode masks
        d_chains = build_simple_chains(sax_data, fg_comps, bitmap)

        # Update trees - see later how to precisely handle that
        l_meta, l_flags = [], []
        for key, chain in d_chains.items():
            (root_id, parent_id, cur_id) = key.split(':')
            ind = fg_comps.get_ind(key)

            # Get children CH xor flag => connex comps
            l_sub_flags = get_mask_cc(
                (sax_ch_data[:, int(ind)].A ^ sax_flags[:, ind].A).reshape(pixel_shape).astype(np.uint8),
                comp_min_size=50
            )

            # Add chain to tree
            d_trees.get(root_id or cur_id, Tree()).create_node(
                f'{parent_id or cur_id}:{cur_id}', cur_id, data=chain,
                parent=parent_id or None
            )

            # Extend meta & flags
            l_meta.extend([
                {'id': f'{root_id or cur_id}:{cur_id}:{FgComponents.rd_id(5)}'}
                for _ in range(len(l_sub_flags))
            ])
            l_flags.extend([csc_matrix(ax_flag.flatten()[:, np.newaxis]) for ax_flag in l_sub_flags])

        # Gather flags
        if l_flags:
            sax_flags = hstack(l_flags)
        else:
            break

    import IPython
    IPython.embed()
    # TODO: visualize the different children by iterating along node of the tree
    bar_plot('Encoding as bar plot', d_chains[0].scales(sparse=True, n_dir=bitmap.nf * 2))

    # Draw shape from encoding (tmp for dev purpose)
    shape_drawer = ShapeDrawer(pixel_shape)
    shape_drawer.draw_shape(d_chains[0])
    return d_trees


def build_simple_chains(sax_data: spmatrix, comps: FgComponents, bitmap: BitMap) -> Dict[str, SimpleChain]:
    # Explode components
    sax_exploded_map = explode(comps.inputs, bitmap, comps.meta)
    sax_exploded_mask = explode(comps.inputs, bitmap, comps.meta, use_bf_mask=True)

    # Build up and down triangulation structure for each direction
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
    d_nodes = build_nodes(bitmap, sax_x_up, comps.meta)
    d_nodes = build_nodes(bitmap, sax_x_down, comps.meta, d_nodes=d_nodes, down=True)

    # Build chains
    return {k: SimpleChain(v) for k, v in d_nodes.items()}


def build_nodes(
        bitmap: BitMap, sax_x: spmatrix, l_meta: List[Dict[str, str]], d_nodes: Dict[int, List[Node]] = None,
        down: bool = False
) -> Dict[int, List[Node]]:

    # Precompute scales and offset coef
    ax_scales = sax_x.sum(axis=0).A[0]
    offset_coef = int(np.cos(np.pi / bitmap.nf) / np.sin(np.pi / bitmap.nf))

    # Init list of nodes if necessary
    if d_nodes is None:
        d_nodes: Dict[str, List[Node]] = {d['id']: [] for d in l_meta}

    # Build nodes for each comp
    for cind, l_sub_inds in groupby(sorted(zip(*sax_x.nonzero()), key=itemgetter(1)), itemgetter(1)):
        l_linds, norm_ind = list(map(itemgetter(0), l_sub_inds)), int((cind % bitmap.nf) + bitmap.nf / 2)
        if down:
            # Get min position
            arg_min = (bitmap.projections[l_linds, norm_ind % bitmap.nf] * ((2 * (norm_ind >= bitmap.nf)) - 1)).argmin()

            # Apply offset in dir and normal
            offset_norm = (((2 * (norm_ind < bitmap.nf)) - 1) * bitmap.transform_op[:, norm_ind % bitmap.nf])
            offset_dir = bitmap.transform_op[:, cind % bitmap.nf]
            p = bitmap.basis[l_linds][arg_min] + (offset_coef * offset_norm) + offset_dir
        else:
            arg_min = (bitmap.projections[l_linds, norm_ind % bitmap.nf] * ((2 * (norm_ind < bitmap.nf)) - 1)).argmin()

            # Apply offset in the inverse direction of normal of current direction
            offset_norm = (((2 * (norm_ind >= bitmap.nf)) - 1) * bitmap.transform_op[:, norm_ind % bitmap.nf])
            offset_dir = bitmap.transform_op[:, cind % bitmap.nf]
            p = bitmap.basis[l_linds][arg_min] + (offset_coef * offset_norm) - offset_dir

        # Set Node
        d_nodes[l_meta[cind // bitmap.nf]['id']].append(Node(
            direction=(cind % bitmap.nf) + bitmap.nf * down, p0=tuple(p.astype(int)), scale=ax_scales[cind]
        ))

    return d_nodes
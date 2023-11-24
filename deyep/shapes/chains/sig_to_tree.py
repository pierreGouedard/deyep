# Global import
from typing import List, Dict, Tuple, Optional, Iterable, Union
import numpy as np
from scipy.sparse import spmatrix, hstack, csc_matrix
from itertools import groupby
from operator import itemgetter
from treelib import Node as TreeNode, Tree

# Local import
from deyep.utils.firing_graph import DeyepFiringGraph
from deyep.linalg.spmat_op import expand, explode
from deyep.models.base import FgComponents, BitMap
from deyep.shapes.drawer import ShapeDrawer
from deyep.utils.plots import bar_plot
from deyep.models.graph import SimpleNode, SimpleChain, WalkableChain, NodeBasis
from deyep.opencv.utils import get_mask_cc


def ax_project(basis: NodeBasis, ax_points: np.array) -> Tuple[np.array, np.array]:
    ax_dir_coef = ax_points.dot(basis.dir) - np.array(basis.p0).dot(basis.dir)
    ax_norm_coef = ax_points.dot(basis.norm) - np.array(basis.p0).dot(basis.norm)
    return ax_dir_coef, ax_norm_coef


def build_walkable_convex_chain_trees(
        sax_flags: spmatrix, sax_data: spmatrix, bitmap: BitMap, pixel_shape: Tuple[int, int],
        debug: bool = False
) -> Dict[str, Tree]:
    """
    Build convex walkable chain part composing a shape. Each convex chain are placed in a tree so to enable
    merging into a unique non convex shape.

    The convex shape part describe the arbitral shape drawn by signals in sax_flags. sax_flags contains
    reshaped binary images, if its dimension in (N, k) it means that pixel_coord[0] * pixel_coord[1] = N and there
    is k binary images to fit.

    the procedure is the following:

    1. For each binary image, get the convex hull (CH)
    2. Represent the CH as a walkable chain
    3. Update a n-ary tree

    At the end of this iteration, create a new set of binary image composed of every
    connexe component of the previously computed CH(S) XOR S, S being the original binary signal.
    Iterate again though the 3 steps.

    The tree is updated in the following way: let a vertex v be a chain that describe the convex hull of a binary
    signal S, if w be a children of v, then w is a chain that describe the convex hull of a connex component of
    S XOR CH(S).

    At the vend of the first iteration, there is as many trees as number of binary signals in passed sax_flags,
    each tree will be composed with a root vertex (first level CH).

    The iteration stops when no more binary signals is added.

    Parameters
    ----------
    sax_flags: sparse matrix, Reshaped binary images
    sax_data: sparse matrix, sparse overcomplete and discretize representation of pixel coordinates
    bitmap: Bitmap used to build pixel data and flags.
    pixel_shape: original size of the binary image.
    debug: bool

    Returns
    -------
    Dict of trees. Each tree provide arranged convex chain that
    """
    # At start, flags are init flags
    l_meta = [{'id': '::' + FgComponents.rd_id(5)} for _ in range(sax_flags.shape[1])]
    d_trees = {d['id'].split(':')[-1]: Tree() for d in l_meta}

    # Until there is no more flag
    while sax_flags.nnz > 0:

        # Create FgComp so to get the convex hull (CH) of flag image
        fg_comps = FgComponents(sax_data.T.dot(sax_flags)).set_levels(bitmap.ndir).set_meta(l_meta)
        sax_ch_data = DeyepFiringGraph.from_comp(fg_comps).seq_propagate(sax_data)

        # Build simple chain of the CH.
        d_chains = build_simple_chains(sax_data, fg_comps, bitmap)

        # Update trees - see later how to precisely handle that
        l_meta, l_flags = [], []
        for key, chain in d_chains.items():
            (root_id, parent_id, cur_id) = key.split(':')
            ind = fg_comps.get_ind(key)

            # Get children: CH xor flag + get list of connex components.
            l_sub_flags = get_mask_cc(
                (sax_ch_data[:, int(ind)].A ^ sax_flags[:, ind].A).reshape(pixel_shape).astype(np.uint8),
                comp_min_size=50
            )

            # Add chain to tree
            d_trees.get(root_id or cur_id, Tree()).create_node(
                f'{parent_id or cur_id}:{cur_id}', cur_id,
                data=build_walkable_chain(chain, bitmap),
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

    # Visualize the different children by iterating along node of the tree
    if debug:
        tree = d_trees[list(d_trees.keys())[0]]
        shape_drawer = ShapeDrawer(pixel_shape)

        # Draw each node
        for node in tree.expand_tree(mode=Tree.WIDTH):
            bar_plot('Encoding as bar plot', tree[node].data.scales(sparse=True, n_dir=bitmap.ndir * 2))

            # Draw shape from encoding (tmp for dev purpose)
            shape_drawer.draw_shape(tree[node].data)

    return d_trees


def build_walkable_chain(chain: SimpleChain, bitmap: BitMap) -> WalkableChain:
    """
    Build walkable chain from simplechain repsenting a Convex Shape.

    The approximation method to get the coordinates of intersection point of the convex hull introduce a bias that
    can be corrected. for each node, we translate the coordinate along its inverse direction, then keep the coordinate
    that give the lower orthogonal projection absolute value coeficient onto previous node normal.

    In addition, a true scale is added to the node, based on usual 2D Euclidean distance and norm and consecutive
    node with equal coordinates are removed.

    Parameters
    ----------
    chain: Simple chain
    bitmap: bitmap

    Returns
    -------

    """
    # Create walkable chain
    wchain = WalkableChain.from_simple_chain(chain.deduplicated(), bitmap)

    # Set maximum pixel offset allowed.
    offset_range = range(int(np.ceil(np.cos(np.pi / bitmap.ndir) / np.sin(np.pi / bitmap.ndir)) + 2))

    # Correct nodes position.
    while not wchain.is_looped:
        # Move current points along its inverse direction
        ax_points = (
                np.array(wchain.curr_node.p0) -
                np.array([i * np.array(wchain.curr_dir()) for i in offset_range])
        ).astype(int)

        # Project candidate points along normal of prev node
        ax_dir, ax_norm = ax_project(wchain.nodes[wchain.cindex(wchain.position - 1)].basis, ax_points)

        # Get closest point along normal of previous node, that has a positive coef along its direction
        ind_closest = np.abs(((ax_dir <= 0) * 100) + ax_norm).argmin()

        # Update curr node position & previous node scale and move to next
        wchain.curr_node.p0 = tuple(ax_points[ind_closest])
        wchain.nodes[wchain.cindex(wchain.position - 1)].scale = ax_dir[ind_closest]
        wchain.next()

    # Reset basis of last node
    wchain.nodes[-1].reset_basis()

    return wchain.deduplicated().init_walk(0, cnt=0)


def build_simple_chains(sax_data: spmatrix, comps: FgComponents, bitmap: BitMap) -> Dict[str, SimpleChain]:

    # Explode components -> Each direction of a firing graph vertex is isolated in own vertex.
    sax_exploded_map = explode(comps.inputs, bitmap, comps.meta)
    sax_exploded_mask = explode(comps.inputs, bitmap, comps.meta, use_bdir_mask=True)

    # Build up and down bit direction' expansion.
    sax_exploded_up = sax_exploded_mask + expand(
        sax_exploded_map, bitmap, keep_only_expanded=True, is_only_up=True
    )
    sax_exploded_down = sax_exploded_mask + expand(
        sax_exploded_map, bitmap, keep_only_expanded=True, is_only_down=True
    )

    # Propagate signal
    sax_x_up = (
        DeyepFiringGraph
        .from_comp(
            FgComponents(sax_exploded_up).set_levels(bitmap.ndir)
        )
        .seq_propagate(sax_data)
    )
    sax_x_down = (
        DeyepFiringGraph
        .from_comp(
            FgComponents(sax_exploded_down).set_levels(bitmap.ndir)
        )
        .seq_propagate(sax_data)
    )

    # Get nodes for each comp
    d_nodes = build_simple_nodes(bitmap, sax_x_up, comps.meta)
    d_nodes = build_simple_nodes(bitmap, sax_x_down, comps.meta, d_nodes=d_nodes, down=True)

    # Build chains
    return {k: SimpleChain(v) for k, v in d_nodes.items()}


def build_simple_nodes(
        bitmap: BitMap, sax_x: spmatrix, l_meta: List[Dict[str, str]], d_nodes: Dict[int, List[SimpleNode]] = None,
        down: bool = False
) -> Dict[int, List[SimpleNode]]:
    # Precompute scales
    ax_scales = sax_x.sum(axis=0).A[0]

    # Init list of nodes if necessary
    if d_nodes is None:
        d_nodes: Dict[str, List[SimpleNode]] = {d['id']: [] for d in l_meta}

    # Build nodes for each comp
    for cind, l_sub_inds in groupby(sorted(zip(*sax_x.nonzero()), key=itemgetter(1)), itemgetter(1)):

        # Get dir and norm ind
        norm_ind = bitmap.norm_ind(cind + ((cind // bitmap.ndir) * bitmap.ndir) + (int(down) * bitmap.ndir))
        l_pix_inds = list(map(itemgetter(0), l_sub_inds))

        # Get min position
        arg_min = bitmap.get_proj(norm_ind, l_pix_inds).argmin()

        # Update SimpleNode dict
        d_nodes[l_meta[cind // bitmap.ndir]['id']].append(SimpleNode(
            direction=(cind % bitmap.ndir) + bitmap.ndir * down,
            p0=tuple(bitmap.pixel_coords[l_pix_inds][arg_min]),
            scale=ax_scales[cind]
        ))

    return d_nodes

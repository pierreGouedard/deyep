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


def merge_simple_chain(l_simple_chain: List[SimpleChain], bitmap: BitMap):
    pass


def reduce_simple_chain_trees(d_chain_tree: Dict[str, Tree], bitmap: BitMap):
    # Todo: parallelize this loop
    for k, tree in d_chain_tree.items():
        # TODO: this can also be separeted start with merging all  level d-1 node of tree,
        #   Then do that again with d-1, the parallelized complexity in log(tree_depth) * (average point in simplechain)
        master_chain = master_update(SimpleChain(), tree.get_node(tree.root), tree)

    return


def master_update(master_chain: SimpleChain, curr_tree_node: TreeNode, tree: Tree):
    print('============ Init =============')
    print(tree)
    print('=========================')

    while not curr_tree_node.data.is_looped:
        # Check for conflicts. Conflicting node can either be a child or a parent
        node, conflicting_tree_node = walk_through_chain_node(
            curr_tree_node.data, [tree.get_node(tnid) for tnid in curr_tree_node.successors(tree.identifier)],
            tree.get_node(curr_tree_node.predecessor(tree.identifier))
        )
        if node.scale > 0:
            master_chain.append(node)

        # If conflicting tree node is a parent of current tree node, then remove curr_tree_node from tree
        if conflicting_tree_node:
            if curr_tree_node.identifier in conflicting_tree_node.successors(tree.identifier):

                import IPython
                IPython.embed()
                tree.remove_node(curr_tree_node.identifier)
                print('============ Child removal =============')
                print(tree)
                print('=========================')

        # TODO: debug area
        print('master_update: New iteration')
        print('============ master_update: New iteration =============')
        print("node to add to master")
        print(node.__dict__)
        print('---------')
        print("master chain:")
        print(master_chain.__dict__)
        print('---------')
        print("current tree node id:")



        shape_drawer = ShapeDrawer((210, 141))
        #import IPython
        #IPython.embed()
        ###

        if conflicting_tree_node is not None:

            print(conflicting_tree_node.identifier)
            print('=========================')

            return master_update(master_chain, conflicting_tree_node, tree)

        # Move to next node in chain
        curr_tree_node.data.next()

        print(curr_tree_node.identifier)
        print('=========================')

    return master_chain


def walk_through_chain_node(
        chain: WalkableChain, child_tree_nodes: Optional[List[TreeNode]] = None,
        parent_tree_node: Optional[TreeNode] = None
) -> Tuple[SimpleNode, Union[TreeNode, None]]:

    # 1. Check for conflict of chain node with children chains in tree
    child_tree_node, init_ind, master_scale = find_children_conflict(chain, child_tree_nodes)

    if child_tree_node is not None:
        # 2. If conflicted, create sub node from current node (node that make the path to conflict)
        master_node = chain.curr_simple_node()
        master_node.scale = master_scale

        # 3. If conflict, process successor node data so that => right order & index start is set
        child_tree_node.data.init_walk(position=init_ind, orient=chain.swapped_orientation)

        return master_node, child_tree_node

    # 6. If no conflict, check for conflict with predecessor node
    parent_tree_node = find_parent_conflict(chain, parent_tree_node)

    if parent_tree_node is not None:
        return chain.curr_simple_node(), parent_tree_node

    # 9. If no conflict, return node unchanged and None as conflicting node
    return chain.curr_simple_node(), None


def find_children_conflict(
        chain: WalkableChain, l_children_tree_nodes: List[TreeNode]
) -> Tuple[Union[TreeNode, None], int, float]:

    # Init. Get curr basis components and
    basis, l_candidates = chain.curr_basis(), []
    for child_tree_node in l_children_tree_nodes:

        # Curr node of chain has a dir and a normal + p0
        for ind, dir_coef, norm_coef in iter_project(basis, child_tree_node.data.points()):
            # TODO: See what is a good tolerance, given how we solve the problem of point's approximation.
            if abs(norm_coef) < 1:
                l_candidates.append((child_tree_node.identifier, ind, dir_coef))

    # TODO: Debug area
    print('============ find_children_conflict =============')
    print(f'curr chain {chain.position}:')
    print('---------')
    print(f'childrens conflicted ({len(l_candidates)}):')
    for (tree_id, node_ind, dir_coef) in l_candidates:
        print(f'child {tree_id}')
        print('---------')
    ####

    if l_candidates:
        # Get candidate closest to curr p0 along the curr direction
        [tree_id, node_ind, dir_coef] = min(l_candidates, key=lambda x: x[-1])

        # Compute new scale of curr_node
        p1_projected, _ = ax_project(basis, np.array(chain.curr_p1())[np.newaxis, :])

        # TODO: we absolutely need to stop this non linear notion of scale.
        #  In addition, if p1_projected ~ dir_coef, we better move current to next.
        master_scale = chain.curr_node.scale * (dir_coef / p1_projected[0])

        print(f'child selected, identifier {tree_id}, position => {node_ind}, master_scale => {master_scale}')
        print('=========================')
        # Recover tree node
        conflicted_tree_node = [tnode for tnode in l_children_tree_nodes if tnode.identifier == tree_id][0]

        return conflicted_tree_node, node_ind, master_scale

    return None, 0, 0


def find_parent_conflict(chain: WalkableChain, parent_tree_node: TreeNode) -> Union[TreeNode, None]:
    # Project curr p1 on parent node and next node.
    ax_curr_dir, ax_curr_norm = ax_project(
        parent_tree_node.data.curr_basis(), np.array(chain.curr_p1())[np.newaxis, :]
    )
    ax_next_dir, ax_next_norm = ax_project(
        parent_tree_node.data.next_basis(), np.array(chain.curr_p1())[np.newaxis, :]
    )

    # Check conflict
    is_conflict, dir_coef = False, None
    if abs(ax_curr_norm[0]) < 1:
        is_conflict, dir_coef = True, ax_curr_dir[0]
    elif abs(ax_next_norm[0]) < 1:
        is_conflict, dir_coef = True, ax_next_dir[0]
        parent_tree_node.data.next()

    ### TODO: Debug area
    print('============ find_parent_conflict =============')
    print(
        f'curr chain at position {chain.position}, p0 => {chain.curr_p1()}, '
        f'p1 => {chain.curr_p1()}, is_conflict => {is_conflict} '
        f'parent_position => {parent_tree_node.data.position}'
    )
    print('---------')
    print('=========================')
    ####

    if is_conflict:
        # Split parent chain curr_node and move to latest
        parent_tree_node.data.split(scale=dir_coef, p0=chain.curr_p1())

        return parent_tree_node

    return None


def iter_project(basis: NodeBasis, l_points: List[Tuple[int, int]]) -> Iterable[Tuple[int, float, float]]:
    # Basis is dir, norm, p0 => always p0 (that change according to order)
    for i, p in enumerate(l_points):
        dir_coef = np.array(p).dot(basis.dir) - np.array(basis.p0).dot(basis.dir)
        norm_coef = np.array(p).dot(basis.norm) - np.array(basis.p0).dot(basis.norm)
        yield i, dir_coef, norm_coef


def ax_project(basis: NodeBasis, ax_points: np.array) -> Tuple[np.array, np.array]:
    ax_dir_coef = ax_points.dot(basis.dir) - np.array(basis.p0).dot(basis.dir)
    ax_norm_coef = ax_points.dot(basis.norm) - np.array(basis.p0).dot(basis.norm)
    return ax_dir_coef, ax_norm_coef


def build_walkable_chain_trees(
        sax_flags: spmatrix, sax_data: spmatrix, bitmap: BitMap, pixel_shape: Tuple[int, int],
        debug: bool = False
) -> Dict[str, Tree]:
    # At start, flags are init flags
    l_meta = [{'id': '::' + FgComponents.rd_id(5)} for _ in range(sax_flags.shape[1])]
    d_trees = {d['id'].split(':')[-1]: Tree() for d in l_meta}

    # Until there is no more flag
    while sax_flags.nnz > 0:

        # Create FgComp of flag's convex hull
        fg_comps = FgComponents(sax_data.T.dot(sax_flags)).set_levels(bitmap.ndir).set_meta(l_meta)
        sax_ch_data = DeyepFiringGraph.from_comp(fg_comps).seq_propagate(sax_data)

        # Encode masks
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

        for node in tree.expand_tree(mode=Tree.WIDTH):
            bar_plot('Encoding as bar plot', tree[node].data.scales(sparse=True, n_dir=bitmap.ndir * 2))

            # Draw shape from encoding (tmp for dev purpose)
            shape_drawer.draw_shape(tree[node].data)

    return d_trees


def build_walkable_chain(chain: SimpleChain, bitmap: BitMap) -> WalkableChain:
    """
    # Todo => can be parallelized (as moving node only on its dir axis)
    Parameters
    ----------
    chain
    bitmap

    Returns
    -------

    """
    # Create walkable chain & set max offset
    wchain = WalkableChain.from_simple_chain(chain, bitmap)
    offset_range = range(int(np.ceil(np.cos(np.pi / bitmap.ndir) / np.sin(np.pi / bitmap.ndir)) + 2))

    # Correct nodes position.
    while not wchain.is_looped:
        # Move current points along its inverse direction
        ax_points = (
                np.array(wchain.curr_node.p0) -
                np.array([i * np.array(wchain.curr_dir()) for i in offset_range])
        ).astype(int)

        # Project candidate points along normal of prev node
        _, ax_norm = ax_project(wchain.nodes[wchain.cindex(wchain.position - 1)].basis, ax_points)

        # update curr node position and move to next
        wchain.curr_node.p0 = tuple(ax_points[np.abs(ax_norm).argmin()])
        wchain.curr_node.reset_basis()
        wchain.next()

    return wchain.init_walk(0)


def build_simple_chains(sax_data: spmatrix, comps: FgComponents, bitmap: BitMap) -> Dict[str, SimpleChain]:
    # Explode components
    sax_exploded_map = explode(comps.inputs, bitmap, comps.meta)
    sax_exploded_mask = explode(comps.inputs, bitmap, comps.meta, use_bdir_mask=True)

    # Build up and down triangulation structure for each direction
    sax_exploded_up = sax_exploded_mask + expand(
        sax_exploded_map, bitmap, keep_only_expanded=True, is_only_up=True
    )
    sax_exploded_down = sax_exploded_mask + expand(
        sax_exploded_map, bitmap, keep_only_expanded=True, is_only_down=True
    )

    # Propagate signal
    sax_x_up = DeyepFiringGraph.from_comp(FgComponents(sax_exploded_up).set_levels(bitmap.ndir))\
        .seq_propagate(sax_data)
    sax_x_down = DeyepFiringGraph.from_comp(FgComponents(sax_exploded_down).set_levels(bitmap.ndir))\
        .seq_propagate(sax_data)

    # Get nodes for each comp
    d_nodes = build_simple_nodes(bitmap, sax_x_up, comps.meta)
    d_nodes = build_simple_nodes(bitmap, sax_x_down, comps.meta, d_nodes=d_nodes, down=True)

    # Build chains
    return {k: SimpleChain(v) for k, v in d_nodes.items()}


def build_simple_nodes(
        bitmap: BitMap, sax_x: spmatrix, l_meta: List[Dict[str, str]], d_nodes: Dict[int, List[SimpleNode]] = None,
        down: bool = False
) -> Dict[int, List[SimpleNode]]:
    # Precompute scales and offset coef
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

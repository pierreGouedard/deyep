# Global import
from typing import List, Dict, Tuple, Iterable, Union
import numpy as np
from treelib import Node as TreeNode, Tree

# Local import
from deyep.models.base import BitMap
from deyep.models.graph import SimpleNode, SimpleChain, WalkableChain, NodeBasis


def merge_simple_chain(l_simple_chain: List[SimpleChain], bitmap: BitMap):
    pass


def build_walkable_chains(d_chain_tree: Dict[str, Tree], bitmap: BitMap):
    # Todo: parallelize this loop
    for k, tree in d_chain_tree.items():
        master_chain = master_update(SimpleChain(), tree.get_node(tree.root), tree)

    return


def master_update(master_chain: SimpleChain, curr_tree_node: TreeNode, tree: Tree, max_iter: int = None):

    # Set automatically max_iter if not set (set it to total number of chain node in entire tree)
    if max_iter is None:
        max_iter = 0
        for tree_node_id in tree.expand_tree():
            max_iter += len(tree.get_node(tree_node_id).data)

    i = 0
    while i < max_iter:
        # Check for conflicts. Conflicting node can either be a child or a parent
        node, conflicting_tree_node = walk_through_chain_node(curr_tree_node, tree)
        if node:
            master_chain.append(node)

        if conflicting_tree_node is not None:
            return master_update(master_chain, conflicting_tree_node, tree)

        if curr_tree_node.data.is_looped:
            break
        # Move to next node in chain
        curr_tree_node.data.next()
        i += 1

    import IPython
    IPython.embed()
    return master_chain


def walk_through_chain_node(
        curr_tree_node: TreeNode, tree: Tree
) -> Tuple[Union[SimpleNode, None], Union[TreeNode, None]]:

    # Get chain of the current tree node and predecessor & sucessor tree nodes
    chain = curr_tree_node.data
    child_tree_nodes = [tree.get_node(tnid) for tnid in curr_tree_node.successors(tree.identifier)]
    parent_tree_node = tree.get_node(curr_tree_node.predecessor(tree.identifier))

    # 1. Check for conflict of chain node with children chains in tree
    child_tree_node, init_ind, master_node = find_children_conflict(chain, child_tree_nodes)

    if child_tree_node is not None:
        # 2. If conflict, process successor node data so that => right order & index start is set
        child_tree_node.data.init_walk(init_ind, orient=chain.swapped_orientation)

        # 3. return master node (to add to master branch) an conflicting tree node
        return master_node, child_tree_node

    # 4. If no child conflict, check for conflict with predecessor node
    parent_tree_node, master_node = find_parent_conflict(chain, parent_tree_node)

    if parent_tree_node is not None:
        # 5. Remove current chain from tree
        tree.remove_node(curr_tree_node.identifier)

        # 6. Return master node (to add to master branch) an conflicting tree node
        return master_node, parent_tree_node

    # 7. If no conflict, return curr node of curr chain updatinf directions and None as conflicting tree node
    master_node = chain.curr_node.copy()
    master_node.direction = chain.curr_direction()
    return master_node.to_simple(), None


def find_children_conflict(
        chain: WalkableChain, l_children_tree_nodes: List[TreeNode]
) -> Tuple[Union[TreeNode, None], int, Union[None, SimpleNode]]:

    # Init. Get curr basis components and
    basis, l_candidates, master_node = chain.curr_basis(), [], None
    for child_tree_node in l_children_tree_nodes:
        # Check for match with current chain node
        l_match_ind = []
        l_distances = [chain.curr_node.dist(p) for p in child_tree_node.data.points()]
        for ind, dist in enumerate(l_distances):
            if abs(dist) < 2:
                l_match_ind.append(ind)

        # Now look at children chain node that lies on the straight line between current chain'
        # current and next node.
        for ind, dir_coef, norm_coef in iter_project(basis, child_tree_node.data.points()):
            if (abs(norm_coef) < 2 and dir_coef > 0) or ind in l_match_ind:
                l_candidates.append((
                    child_tree_node.identifier, ind, ind in l_match_ind,
                    abs(dir_coef), norm_coef
                ))

    if l_candidates:
        # If any candidate matching current chain node position, then return any of those candidates
        # as the conflicting node and do not return current node of current chain as master node
        # (to add to master chain).
        l_match = list(filter(lambda x: x[2], l_candidates))
        if l_match:
            conflicted_tree_node = [
                tnode for tnode in l_children_tree_nodes if tnode.identifier == l_match[0][0]
            ][0]
            return conflicted_tree_node, l_match[0][1], None

        # Get candidate closest to curr p0 along the curr direction of curr chain
        [tree_id, node_ind, _, dir_coef, _] = min(l_candidates, key=lambda x: x[-2])

        # If conflicting tree node is localized farther than curr chain's next node, or is very close
        # to curr chain next node, ignore conflict.
        if chain.curr_scale() < dir_coef + 1.5:
            return None, 0, None

        # Recover tree node
        conflicted_tree_node = [tnode for tnode in l_children_tree_nodes if tnode.identifier == tree_id][0]
        master_node = chain.curr_node.copy()
        master_node.direction = chain.curr_direction()

        return conflicted_tree_node, node_ind, master_node.to_simple()

    return None, 0, None


def find_parent_conflict(
        chain: WalkableChain, parent_tree_node: TreeNode
) -> Union[Tuple[Union[TreeNode, None], Union[SimpleNode, None]]]:

    # If no parent tree skip
    if parent_tree_node is None:
        return None, None

    # Check if current node match node of parent chain.
    parent_chain, is_match, is_conflict, dir_incr = parent_tree_node.data.copy(), False, False, 0
    while True:

        # Compute distance between parent chain next node coordinates and curr chain node coordinates
        # If close enough and at least 1 edge has been explored on current chain, claim a match with parent
        # current node
        if chain.curr_node.dist(parent_chain.curr_p0()) < 2. and chain.counter_value > 0:
            is_match = True
            break

        if dir_incr > 1:
            break

        # Now look if curr chain node lies on the straight line between parent chain'
        # current and next node.
        (ind, dir_coef, norm_coef) = list(iter_project(parent_chain.curr_basis(), [chain.curr_node.p0]))[0]

        # If conflicting tree node is localized farther or too close than parent chain's next node,
        # ignore conflict.
        if parent_chain.curr_scale() < dir_coef + 1.5:
            parent_chain.next()
            continue

        # If valid conflict, skip.
        if abs(norm_coef) < 2 and chain.counter_value > 0:
            is_conflict = True
            break

        # Keep track of current basis
        curr_direction = parent_chain.curr_direction()

        # Increment parent chain
        parent_chain.next()

        # Get new direction and increment direction increment since start
        next_direction = parent_chain.curr_direction()
        dir_incr += direction_diff(curr_direction, next_direction, parent_chain.ndir)

    if is_match:
        parent_tree_node.data = parent_chain
        return parent_tree_node, None

    elif is_conflict:
        # Keep curr node of current chain as master node and update its direction
        master_node = chain.curr_node
        master_node.direction = parent_chain.curr_direction()

        # As master_node will be returned, move to next node of parent chain.
        parent_chain.next()

        # replace parent_tree_node data with parent_chain
        parent_tree_node.data = parent_chain

        return parent_tree_node, master_node.to_simple()

    return None, None


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


def direction_diff(dir_a: int, dir_b: int, ndir: int) -> int:
    return min([
            abs(dir_a - dir_b),
            abs(((dir_a + ndir) % (2 * ndir)) - ((dir_b + ndir) % (2 * ndir)))
        ])

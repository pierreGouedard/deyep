# Global import
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

import numpy as np

# Local import
from deyep.models.base import BitMap


@dataclass
class SimpleNode:
    """Data structure that store base data of a node.

    A node is essentially a pixel in theimage that point in a certain direction at a given intensity.

    The node has a parameter p0 that provide the euclidean coordinate in the image of the node.

    The direction is an integer that indicate in which direction point the node in the image.

    The scale represent the norm of the direction vector.

    The index of the direction, together with the scale, may be used to get the 2D euclidean coorinate
    of the direction vector in the image - if a bitmap is available.
    """
    direction: int
    p0: Tuple[int, int]
    scale: float

    def copy(
            self, direction: Optional[str] = None, p0: Optional[Tuple[int, int]] = None,
            scale: Optional[float] = None
    ) -> 'SimpleNode':
        return SimpleNode(direction=direction or self.direction, p0=p0 or self.p0, scale=scale or self.scale)


@dataclass
class SimpleChain:
    """Data structure that store base data of a simple chain.

    The simple chain is a chain of simple nodes. It essentially consists in a list of SimpleNodes.

    It also implements utils functionality to extract nodes proprieties.
    """
    nodes: List[SimpleNode]

    def __init__(self, nodes: Optional[List[SimpleNode]] = None):
        self.nodes = nodes if nodes is not None else []

    def __len__(self):
        return len(self.nodes)

    def cindex(self, index) -> int:
        return (index + len(self)) % len(self)

    def directions(self) -> np.array:
        return np.array([n.direction for n in self.nodes], dtype=int)

    def scales(self, sparse: bool = False, n_dir: int = None) -> np.array:
        if sparse:
            ax_scales = np.zeros(n_dir)
            ax_scales[self.directions()] = self.scales()
            return ax_scales

        return np.array([n.scale for n in self.nodes])

    def points(self) -> np.array:
        return np.stack([np.array(n.p0) for n in self.nodes])

    def append(self, node: SimpleNode):
        self.nodes.append(node)

    def deduplicated(self) -> 'SimpleChain':
        deduplicated_nodes, i = [self.nodes[0]], 0
        for j in range(1, len(self)):
            if deduplicated_nodes[i].p0 != self.nodes[j].p0:
                deduplicated_nodes.append(self.nodes[j])
                i += 1
            else:
                deduplicated_nodes[i] = self.nodes[j]

        self.nodes = deduplicated_nodes

        return self



@dataclass
class NodeBasis:
    """Data structure that store data of a node basis.

    A node basis is a complete 2D basis, with a pair of orthogonal vector coordinates
    (in the usual 2D Euclidean space) and an origin.

    """
    p0: Tuple[int, int]
    dir: Tuple[float, float]
    norm: Tuple[float, float]

    def angle(self, point: Union[Tuple[int, int], np.ndarray]):
        # Get point as numpy array
        ax_point = point if isinstance(point, np.ndarray) else np.array(point)

        return self.rel_angle(ax_point - np.array(self.p0))

    def rel_angle(self, rel_point: np.ndarray):
        # Get point as numpy array
        ax_rel_point = rel_point if isinstance(rel_point, np.ndarray) else np.array(rel_point)

        # compute distance on node basis
        return np.arctan(
            ax_rel_point.dot(self.norm) / (ax_rel_point.dot(self.dir)+ 1e-6)
        )


@dataclass
class DirectedNode(SimpleNode):
    """Data structure that store data of a directed node.

    The Directed Node inherit from Simple node properties and provide additional information.

    dir: The direction, as a 2D coordinate in the image euclidean space.
    norm: The normal to the direction, as a 2D coordinate in the image euclidean space.

    """
    dir: Tuple[float, float]
    norm: Tuple[float, float]
    _opposite_dir: Optional[Tuple[float, float]] = None
    _basis: Optional[NodeBasis] = None

    @property
    def opposite_dir(self) -> Tuple[float, float]:
        if self._opposite_dir is None:
            self._opposite_dir = (-self.dir[0], -self.dir[1])
        return self._opposite_dir

    @property
    def basis(self):
        if self._basis is None:
            self._basis = NodeBasis(
                p0=self.p0, dir=self.dir, norm=self.norm
            )
        return self._basis

    def reset_basis(self):
        self._basis = None

    def copy(self, **kwargs):
        return DirectedNode(
            dir=(self.dir[0], self.dir[1]), norm=(self.norm[0], self.norm[1]),
            **{'p0': self.p0, 'direction': self.direction, 'scale': self.scale, **kwargs}
        )

    def to_simple(self):
        return super().copy()

    def dist(self, point: Union[Tuple[int, int], np.ndarray]) -> float:

        # Get point as numpy array
        ax_point = point if isinstance(point, np.ndarray) else np.array(point)

        # compute distance on node basis
        dist = np.linalg.norm(ax_point - np.array(self.p0))

        return dist

    def project(self, point: Union[Tuple[int, int], np.ndarray]):

        # Get point as numpy array
        ax_point = point if isinstance(point, np.ndarray) else np.array(point)

        # Project on node basis
        ax_dir_coef = ax_point.dot(self.dir) - np.array(self.p0).dot(self.dir)
        ax_norm_coef = ax_point.dot(self.norm) - np.array(self.p0).dot(self.norm)

        return ax_dir_coef, ax_norm_coef

    def project_many(self, points: Union[List[Tuple[int, int]], np.ndarray]):
        pass

@dataclass
class WalkableChain(SimpleChain):
    """Data structure that store data of a WalkableChain.

    The walkable chain inherit from a Simple chain. it is composed of a list of DirectedNode,
    an orientation and a current position.

    The orientation describes the way the chain is walked through, if set to 'trigo' the node of the list will be
    visited incrementally by index ascendant order. If set to 'anti-trigo' the list of node will be visited by
    index descendant order.

    While SimpleNodes composing the simple chain have a direction that is an integer index that refers to a direction
    of the bitmap, DirectedNodes that compose the Walkable Chain has an independant concept of direction and norm
    that differs from bitmap' concept of direction and norm. The direction of a DirectedNode build from a SimpleNode
    with direction integer 0 <= i <= ndir is equal to the bitmap's normal of bitmap's direction i, its norm
    is equal to the bitmap's direction i.

    The DirectedNodes have a fixed definition of direction and norm that is perfectly fitted for a 'trigo' way walk
    through the chain. Indeed, at trigo orientation, at position 0 <= i <= nb nodes, the direction of the
    current DirectedNode placed at its coordinates, point toward the coordinate of DirectedNode at index i + 1
    and the Norm is the usual norm of that direction. Furthermore the next() method will place us at position i + 1
    that correspond to the next directed node coordinates.

    Yet, when walking through the chain in a 'anti-trigo' way, the walkable chain needs to handle:
     * Walking throught the chain consist in decrementing position instead of incrementing it
     * Correct the direction of each Directed node so that the direction point to the previous position in the list
       instead the of the next one.

    For ease of implementation, the 'anti-trigo' walk through of the walkable chain introduce a shift in the indexing.
    At position 0 <= i <= nb nodes, in the 'anti-trigo' orientation, we consider that we currently reached the
    coordinates of directed node at position i + 1, and the next() method will place at the coordinates of node at
    i, while the walkable chain position will be decrement to i-1.

    With this logic, at position i, 'anti-trigo' orientation, the corrected current direction is the opposited
    direction of the DirectedNode direction at position i and when placed at coordinates of Directed node at position i+1, it
    points toward the coordinate of DirectedNode at position i. The norm is the unchanged norm of Directed
    node norm at position i.

    """
    nodes: List[DirectedNode]
    _orient: str
    ndir: int
    position: int = 0
    _cnt: int = 0


    def __init__(
            self, ndir: int, directed_nodes: List[DirectedNode]
    ):
        self._orient = 'trigo'
        self.ndir = ndir
        super(WalkableChain, self).__init__(directed_nodes)

    @staticmethod
    def from_simple_chain(simple_chain: SimpleChain, bitmap: BitMap):
        # The directed node dir & norm are respectively the bitmap norm & dir
        return WalkableChain(
            ndir=bitmap.ndir,
            directed_nodes=[
                DirectedNode(
                    dir=tuple(bitmap.get_norm_coords(dir_ind=node.direction)),
                    norm=tuple(bitmap.get_dir_coords(node.direction)),
                    **node.__dict__
                )
                for node in simple_chain.nodes
            ]
        )

    @property
    def orient(self):
        return self._orient

    @property
    def is_trigo(self):
        return self._orient == 'trigo'

    @property
    def min_angle(self):
        return np.pi / self.ndir

    @property
    def swapped_orientation(self):
        return 'trigo' if self.orient == 'anti-trigo' else 'anti-trigo'

    @property
    def is_looped(self):
        return self._cnt >= len(self)

    @property
    def counter_value(self):
        return self._cnt

    @property
    def curr_node(self) -> DirectedNode:
        return self.nodes[self.position] if self._orient == 'trigo' else self._get_prev_node()

    def copy(self):
        return (
            WalkableChain(ndir=self.ndir, directed_nodes=[n.copy() for n in self.nodes])
            .init_walk(
                position=self.position, orient=self.orient,
                is_trigo_position=self.orient == "trigo", cnt=self._cnt
            )
        )

    def __len__(self):
        return len(self.nodes)

    def init_walk(
            self, position: int, orient: Optional[str] = None,
            is_trigo_position: Optional[bool] = True, cnt: Optional[int] = None
    ) -> 'WalkableChain':

        # Set orientation
        self._orient = orient or self._orient

        # Process position according to orientation of passed position.
        self.position = self.cindex(position)
        if self._orient  != 'trigo' and is_trigo_position:
            self.position = self.cindex(position - 1)
        elif self._orient  == 'trigo' and not is_trigo_position:
            self.position = self.cindex(position + 1)

        # Init count
        self._cnt = cnt or self._cnt

        return self

    def add_next_node(
            self, new_node: DirectedNode, move_to_next: bool = False, min_scale: float = 1.5
    ):
        # Get current node
        curr_node = self.nodes[self.position]

        # Project new node onto current node basis
        dir_coef, _ = curr_node.project(new_node.p0)
        diff_scale = curr_node.scale - max(dir_coef, 0)

        # If director coefficient is too low (< -min_scale) or too large compared to , raise error as it means
        # the new node coordinate doesn't lies between chain's curr and next node - it lies before curr node.
        # If the scale is below min_scale, ignore new node.
        if (dir_coef < -min_scale) or (diff_scale < -min_scale):
            raise ValueError(
                'Node that is attempted to add is not placed between current and next node of the chain.'
            )

        # Update new node' dir & norm
        new_node.dir, new_node.norm = curr_node.dir, curr_node.norm

        # Update node list
        self.nodes = [
            *self.nodes[: max(self.position, 0)],
            *[curr_node, new_node],
            *self.nodes[min(self.position + 1, len(self) - 1):]
        ]

        # Update position (for anti trigo oriented chain)
        if self.orient == 'anti-trigo':
            self.position += 1

        if move_to_next:
            self.next()

    def next(self):
        if self.orient == 'trigo':
            self.position = self.cindex(self.position + 1)
        else:
            self.position = self.cindex(self.position - 1)

        self._cnt += 1

    def next_to(self, position: int) -> None:
        while position != self.position:
            self.next()

    def _get_next_node(self) -> DirectedNode:
        if self.orient == 'trigo':
            return self.nodes[self.cindex(self.position + 1)]
        else:
            return self.nodes[self.cindex(self.position - 1)]

    def _get_prev_node(self) -> DirectedNode:
        if self.orient == 'trigo':
            return self.nodes[self.cindex(self.position - 1)]
        else:
            return self.nodes[self.cindex(self.position + 1)]

    def curr_dir(self) -> Tuple[float, float]:
        if self.orient == 'trigo':
            return self.nodes[self.cindex(self.position)].dir
        else:
            return self.nodes[self.cindex(self.position)].opposite_dir

    def curr_norm(self) -> Tuple[float, float]:
        return self.nodes[self.cindex(self.position)].norm

    def curr_direction(self) -> int:
        if self.orient == 'trigo':
            return self.nodes[self.cindex(self.position)].direction % (2 * self.ndir)
        else:
            return (self.nodes[self.cindex(self.position)].direction + self.ndir) % (2 * self.ndir)

    def curr_p0(self) -> Tuple[int, int]:
        if self.orient == 'trigo':
            return self.nodes[self.cindex(self.position)].p0
        else:
            return self._get_prev_node().p0

    def curr_p1(self) -> Tuple[int, int]:
        if self.orient == 'trigo':
            return self._get_next_node().p0
        else:
            return self.nodes[self.cindex(self.position)].p0

    def curr_scale(self) -> float:
        return self.nodes[self.cindex(self.position)].scale

    def curr_basis(self) -> NodeBasis:
        return NodeBasis(
            p0=self.curr_p0(), dir=self.curr_dir(), norm=self.curr_norm()
        )


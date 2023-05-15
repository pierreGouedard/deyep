# Global import
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np

# Local import
from deyep.models.base import BitMap


@dataclass
class SimpleNode:
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
    nodes: List[SimpleNode]

    def __init__(self, nodes: Optional[List[SimpleNode]] = None):
        self.nodes = nodes if nodes is not None else []

    def __len__(self):
        return len(self.nodes)

    def cindex(self, index) -> int:
        return (index + len(self)) % len(self)

    def scales(self, sparse: bool = False, n_dir: int = None) -> np.array:
        if sparse:
            ax_scales = np.zeros(n_dir)
            ax_scales[self.directions()] = self.scales()
            return ax_scales

        return np.array([n.scale for n in self.nodes])

    def points(self) -> np.array:
        return np.stack([np.array(n.p0) for n in self.nodes])

    def directions(self) -> np.array:
        return np.array([n.direction for n in self.nodes], dtype=int)

    def append(self, node: SimpleNode):
        self.nodes.append(node)


@dataclass
class NodeBasis:
    p0: Tuple[int, int]
    dir: Tuple[float, float]
    norm: Tuple[float, float]


@dataclass
class DirectedNode(SimpleNode):
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


@dataclass
class WalkableChain(SimpleChain):

    nodes: List[DirectedNode]
    _orient: str
    position: int = 0
    _cnt: int = 0

    def __init__(
            self, simple_nodes: Optional[List[SimpleNode]] = None,
            directed_nodes: Optional[List[DirectedNode]] = None
    ):
        self._orient = 'trigo'
        super(WalkableChain, self).__init__(simple_nodes or directed_nodes or [])

    @staticmethod
    def from_simple_chain(simple_chain: SimpleChain, bitmap: BitMap):
        # The directed node dir & norm are respectively the bitmap norm & dir
        return WalkableChain([
            DirectedNode(
                dir=tuple(bitmap.get_norm_coords(dir_ind=node.direction)),
                norm=tuple(bitmap.get_dir_coords(node.direction)),
                **node.__dict__
            )
            for node in simple_chain.nodes
        ])

    @property
    def orient(self):
        return self._orient

    @property
    def swapped_orientation(self):
        return 'trigo' if self.orient == 'anti-trigo' else 'anti-trigo'

    @property
    def is_looped(self):
        return self._cnt // len(self) >= 1

    @property
    def curr_node(self) -> SimpleNode:
        return self.nodes[self.position]

    def __len__(self):
        return len(self.nodes)

    def next(self):
        if self.orient == 'trigo':
            self.position = self.cindex(self.position + 1)
        else:
            self.position = self.cindex(self.position - 1)

        self._cnt += 1

    def next_to(self, position: int) -> None:
        while position != self.position:
            self.next()

    def init_walk(self, position: int, orient: Optional[str] = None) -> 'WalkableChain':
        self._orient = orient or self._orient
        self.position = self.cindex(self.position - 1 if self._orient == 'anti-trigo' else position)
        self._cnt = 0

        return self

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

    def curr_basis(self) -> NodeBasis:
        return NodeBasis(
            p0=self.curr_p0(), dir=self.curr_dir(), norm=self.curr_norm()
        )
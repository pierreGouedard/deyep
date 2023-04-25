# Global import
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np

# Local import


@dataclass
class SimpleNode:
    direction: int
    p0: Tuple[int, int]
    scale: float

    def copy(
            self, direction: Optional[str] = None, p0: Optional[Tuple[int, int]] = None,
            scale: Optional[float] = None
    ) -> 'Node':
        return SimpleNode(direction=direction or self.direction, p0=p0 or self.p0, scale=scale or self.scale)


@dataclass
class DirectedNode:
    simple_node: SimpleNode
    dir: Tuple[float, float]
    norm: Tuple[float, float]


@dataclass
class SimpleChain:
    nodes: List[SimpleNode]

    def __init__(self, l_nodes: Optional[List[SimpleNode]] = None):
        self.nodes = l_nodes if l_nodes is not None else []

    def __len__(self):
        return len(self.nodes)

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
class WalkableChain:

    simple_chain: SimpleChain
    _orient: str
    position: int = 0
    _cnt: int = 0

    def __init__(self, simple_chain: Optional[SimpleChain] = None):
        self.simple_chain = simple_chain if simple_chain is not None else SimpleChain()
        self._orient = 'trigo'

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
        return self.simple_chain.nodes[self.position]

    def __len__(self):
        return len(self.simple_chain)

    def next(self):
        if self.orient == 'trigo':
            self.position = (self.position + 1) % len(self)
        else:
            self.position = (len(self) + (self.position - 1)) % len(self)

        self._cnt += 1

    def init_walk(self, position: int, orient: str):
        self.position = position
        self._orient = orient
        self._cnt = 0

    def _get_next_node(self):
        if self.orient == 'trigo':
            return self.simple_chain.nodes[(self.position + 1) % len(self)]
        else:
            return (len(self) + (self.position - 1)) % len(self)

    def _get_prev_node(self):
        if self.orient == 'trigo':
            return self.simple_chain.nodes[(len(self) + (self.position - 1)) % len(self)]
        else:
            return self.simple_chain.nodes[(self.position + 1) % len(self)]

    def curr_dir(self) -> Tuple[float, float]:
        pass

    def curr_norm(self) -> Tuple[float, float]:
        pass

    def curr_p0(self) -> Tuple[int, int]:
        if self.orient == 'trigo':
            return self.simple_chain.nodes[self.position % len(self)].p0
        else:
            return self._get_prev_node().p0

    def curr_p1(self) -> Tuple[int, int]:
        if self.orient == 'trigo':
            return self._get_next_node().p0
        else:
            return self.simple_chain.nodes[self.position % len(self)].p0


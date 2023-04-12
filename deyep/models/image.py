# Global import
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np

# Local import


@dataclass
class Node:
    direction: int
    p0: Tuple[int, int]
    scale: float


@dataclass
class SimpleChain:
    nodes: List[Node]
    orient: str

    def __init__(self, l_nodes: Optional[List[Node]] = None, orient: Optional[str] = 'trigo'):
        self.nodes = l_nodes if l_nodes is not None else []
        self.orient = orient

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

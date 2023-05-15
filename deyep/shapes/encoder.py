# Global import
import pickle
from typing import List
import numpy as np


# Local import
from deyep.linalg.discretizer import ImgCoordDiscretizer
from deyep.models.base import BitMap
from deyep.models.graph import SimpleChain, WalkableChain
from deyep.shapes.chain_op import build_walkable_chain_trees, reduce_simple_chain_trees, merge_simple_chain


class ImageEncoder:
    # TODO Last piece to add is the open cv segmenter to generate masks
    def __init__(self, l_masks: List[np.array], n_features: int):
        self.pixel_shape = l_masks[0].shape[:2]
        self.masks = l_masks

        # Discretize images
        self.data, self.flags, self.bitmap = self.discretize_image(n_features)

    def discretize_image(self, n_features):
        # Encode pixels coordinates & mask image
        img_discretizer = ImgCoordDiscretizer(max(self.pixel_shape), n_features, 1, 0.0).fit(self.masks[0])

        # Encode labels
        data = img_discretizer.transform(img_discretizer.augmented_coords)
        flag = img_discretizer.transform_mask_labels(self.masks)

        # Set bitmap
        bitmap = BitMap(img_discretizer.bd_map, img_discretizer.pixel_coords, img_discretizer.transformation)

        return data, flag, bitmap

    def encode_image(self) -> SimpleChain:
        # Build chain tree for each masks
        # d_chain_trees = build_walkable_chain_trees(
        #     self.flags, self.data, self.bitmap, self.pixel_shape, debug=True
        # )

        # TODO: save chain_tree for smoother dev
        # with open('test_chain_tree.pckl', 'wb') as handle:
        #    pickle.dump(d_chain_trees, handle)

        with open('test_chain_tree.pckl', 'rb') as handle:
            d_chain_trees = pickle.load(handle)

        # Reduce Chain Trees
        l_chains = reduce_simple_chain_trees(d_chain_trees, self.bitmap)

        # Merge all chains
        chain = merge_simple_chain(l_chains, self.bitmap)

        return chain

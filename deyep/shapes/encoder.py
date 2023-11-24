# Global import
from typing import List
import numpy as np
import pickle

# Local import
from deyep.linalg.discretizer import ImgCoordDiscretizer
from deyep.models.base import BitMap
from deyep.models.graph import SimpleChain
from deyep.shapes.chains.tree_to_chain import build_walkable_chains, merge_simple_chain
from deyep.shapes.chains.sig_to_tree import build_walkable_convex_chain_trees

class ImageEncoder:
    """
    Encode binary images (refered as masks) as a walkable edge graph.

    """
    # TODO Last piece to add is the open cv segmenter to generate masks
    def __init__(self, l_masks: List[np.array], n_dirs: int):
        self.pixel_shape = l_masks[0].shape[:2]
        self.masks = l_masks

        # Discretize images
        self.data, self.flags, self.bitmap = self.discretize_image(n_dirs)

    def discretize_image(self, n_dirs: int):
        """
        Transform image pixel into a sparse representation

        The sparse representation of an image is a representation where each image pixel is represented as
        discretized and sparsed coordinates in an over complete basis in the 2D euclidean space.

        If an image of size (n_row, n_col) is represented using n_dir directions (overcomplete basis of n_dir basis),
        it will be represented as a sparse boolean matrix of size (n_row * n_col, X) where each row has n_dir
        non False values. Where X = [n_bin * (n_dir - 1)] + min([nbin, n_row, n_col]).

        More details on the transformation in `ImgCoordDiscretizer`

        It returns a tuple composed of:
            - Sparse coordinate representation of image
            - Reshaped masks (from list of n * (n_row, n_cols) binary array to a sparse binary array of
                shape (n_row * n_col, n)
            - bitmap: Abstraction that enable useful data & operation on discretized & sparse representation
                of the image

        Parameters
        ----------
        n_dirs: number of direction to use in to encode image

        """
        # Encode pixels coordinates & mask image
        img_discretizer = ImgCoordDiscretizer(max(self.pixel_shape), n_dirs, 1, 0.0).fit(self.masks[0])

        # Encode labels
        data = img_discretizer.transform(img_discretizer.augmented_coords)
        flag = img_discretizer.transform_mask_labels(self.masks)

        # Set bitmap
        bitmap = BitMap(img_discretizer.bitdir_map, img_discretizer.pixel_coords, img_discretizer.transformation)

        return data, flag, bitmap

    def encode_image(self) -> SimpleChain:
        # Build chain tree for each masks
        # d_chain_trees = build_walkable_convex_chain_trees(
        #     self.flags, self.data, self.bitmap, self.pixel_shape, debug=True
        # )

        # # TODO: save chain_tree for smoother dev
        # with open('test_chain_tree.pckl', 'wb') as handle:
        #    pickle.dump(d_chain_trees, handle)

        # Load tree
        with open('test_chain_tree.pckl', 'rb') as handle:
            d_chain_trees = pickle.load(handle)

        # Reduce Chain Trees
        l_chains = build_walkable_chains(d_chain_trees, self.bitmap)

        # Merge all chains
        chain = merge_simple_chain(l_chains, self.bitmap)

        return chain

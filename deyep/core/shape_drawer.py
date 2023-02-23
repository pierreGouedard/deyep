# Global import
from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse import lil_matrix, hstack

# Local import
from deyep.core.spmat_op import expand, explode, fill_gap


class ShapeDrawer:
    def __init__(self, bitmap, im_shape, factor=0.5):
        self.bitmap = bitmap
        self.im_shape = im_shape
        self.factor = 3

    def draw_shape(self, img_encoder, ax_codes: np.ndarray):

        n_dir = img_encoder.n_directions
        # Init right most point
        l_X = [
            np.hstack([
                np.ones([ax_codes.shape[0], 1]) * int(self.im_shape[0] / 2),
                np.ones([ax_codes.shape[0], 1]) * int(self.im_shape[1] / 2)
            ])
        ]
        #
        for i in range(n_dir * 2):
            ax_transform = np.array([-np.cos(i * np.pi / n_dir), -np.sin(i * np.pi / n_dir)])
            l_X.append(l_X[-1] + self.factor * np.sqrt(ax_codes[:, [i]]) * ax_transform)

        ax_points = np.vstack(l_X[1:])
        X = img_encoder.transform(ax_points, transform_input=True)
        n_verts = ax_codes.shape[0]

        # Denest vertices
        sax_denesting = lil_matrix((n_verts, n_verts * n_dir * 2))
        sax_denesting[
            sum([[k for i in range(n_verts * n_dir * 2) if i % n_verts == k] for k in range(n_verts)], []),
            sum([[i for i in range(n_verts * n_dir * 2) if i % n_verts == k] for k in range(n_verts)], []),
        ] = True

        # fill gap of the shit below
        sax_i = fill_gap(sax_denesting.dot(X).T, self.bitmap)

        return sax_i
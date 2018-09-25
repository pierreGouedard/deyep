# Global import

# Local import


def init_imputer(imputer):

    imputer.read_raw_data('forward.npz', 'backward.npz')
    imputer.run_preprocessing()
    imputer.write_features('forward.npz', 'backward.npz')

    return imputer
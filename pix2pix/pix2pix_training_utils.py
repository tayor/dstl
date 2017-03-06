
#
# pix2pix training utils
# Code inspired by https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/pix2pix
#

import numpy as np


def compute_disc_patches(X_in, X_out,
                       generator_model,
                       batch_counter,
                       patch_size,
                       label_smoothing=False,
                       label_flipping=0):
    """
    Method computes data for the descriminator, alternating real and generated images
    """

    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Produce an output
        X_disc = generator_model.predict(X_in)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        X_disc = X_out
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Now extract patches form X_disc
    X_disc = extract_patches(X_disc, patch_size)

    return X_disc, y_disc


def extract_patches(X, patch_size):

    # Now extract patches form X_disc
    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[2] / patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[3] / patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, :, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]])
    return list_X

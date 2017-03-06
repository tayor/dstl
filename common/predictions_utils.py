
import os
import sys
sys.path.append("../common")

import numpy as np
import cv2

from data_utils import get_filename
from preprocessing_utils import create_pan_rad_inds_ms
from geo_utils.GeoImageTilers import GeoImageTilerConstSize
from geo_utils.GeoImage import from_ndarray, GeoImage
from training_utils import normalize_image


def compute_predictions(image_id, model, channels=None, image_type='input', tile_size = (256, 256), mean_image=None, std_image=None, batch_size=4, scale=1, out_size=(3500, 3500)):
    """
    Method to compute predictions using trained model
    :param image_id: input image id on which compute predictions
    :param model: trained model
    :param mean_image: to normalize from input image
    :param std_image: to normalize from input image
    :param batch_size: number of tiles from input image on which we run model.predict_on_batch
    :param out_size: output predictions image size
    :return: ndarray with predictions
    """
    Y_predictions = np.zeros(out_size + (model.output_shape[1], ), dtype=np.float32)
    
    overlapping = int(min(tile_size[0], tile_size[1]) * 0.25)

    if mean_image is not None and std_image is not None:
        if channels is not None:
            mean_image = mean_image[:, :, channels]
            std_image = std_image[:, :, channels]

    image_filename = get_filename(image_id, image_type)
    if not os.path.exists(image_filename):
        x = create_pan_rad_inds_ms(image_id, remove_generated_files=True)
        gimg = from_ndarray(x)
    else:
        gimg = GeoImage(image_filename)

    tiles = GeoImageTilerConstSize(gimg, tile_size, overlapping, scale=scale)
    count = 0
    X = np.zeros((batch_size, model.input_shape[1], tile_size[1], tile_size[0]), dtype=np.float32)
    xs = []
    ys = []
    for tile, x, y in tiles:

        if channels is not None:
            tile = tile[:,:,channels]

        tile_size_s = (tile_size[0] * scale, tile_size[1] * scale)

        if mean_image is not None and std_image is not None:
            mean_tile_image = mean_image[y:y + tile_size_s[1], x:x + tile_size_s[0], :]
            std_tile_image = std_image[y:y + tile_size_s[1], x:x + tile_size_s[0], :]
            if scale > 1:
                mean_tile_image = cv2.resize(mean_tile_image, dsize=tile_size, interpolation=cv2.INTER_LINEAR)
                std_tile_image = cv2.resize(std_tile_image, dsize=tile_size, interpolation=cv2.INTER_LINEAR)

            tile = normalize_image(tile, mean_tile_image, std_tile_image)

        X[count, :, :, :] = tile.transpose([2, 0, 1])
        xs.append(x)
        ys.append(y)
        count += 1

        if count == batch_size:
            Y_pred = model.predict_on_batch(X)
            for i in range(batch_size):
                img = Y_pred[i, :, :, :].transpose([1, 2, 0])
                if scale > 1:
                    img = cv2.resize(img, dsize=tile_size_s, interpolation=cv2.INTER_LINEAR)
                Y_predictions[ys[i]:ys[i] + tile_size_s[1], xs[i]:xs[i] + tile_size_s[0], :] = img
            count = 0
            xs = []
            ys = []
    return Y_predictions
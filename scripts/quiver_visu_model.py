
import numpy as np

import quiver_engine.server as qs

import sys
sys.path.append("../common/")

from data_utils import LABELS
from image_utils import TRAIN_DATA
from unet import unet_zero


tile_size = (256, 256)
channels = np.array([0, 1, 2, 3, 5, 6, 7, 8])
n_channels = len(channels)
labels = np.array([0, 5, 6, 2])
n_labels = len(labels)

model = unet_zero(n_labels, n_channels, *tile_size, deep=False, n_filters_0=16)

qs.launch(model, classes=np.array(LABELS)[labels].tolist(), input_folder=TRAIN_DATA)

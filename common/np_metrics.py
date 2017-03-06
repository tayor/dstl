
import numpy as np


def score(y_true, y_pred):
    smooth = 1e-12
    if len(y_true.shape) == 2:
        y_true = y_true[:,:,None]
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:,:,None]
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    intersection = np.sum(y_true * y_pred, axis=(0, 1))
    sum_ = np.sum(y_true + y_pred, axis=(0, 1))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return np.mean(jac)
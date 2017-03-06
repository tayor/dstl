
#
# Helper method to perform a post-processing on predicted data
#
import numpy as np
import cv2

import sys
sys.path.append("../common/")
from data_utils import LABELS


def sieve(image, size):
    """
    Filter removes small objects of 'size' from binary image
    Input image should be a single band image of type np.uint8

    Idea : use Opencv findContours
    """
    assert image.dtype == np.uint8, "Input should be a Numpy array of type np.uint8"

    sq_limit = size**2
    lin_limit = size*4

    out_image = image.copy()
    image, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if hierarchy is not None and len(hierarchy) > 0:
        hierarchy = hierarchy[0]
        index = 0
        while index >= 0:
            contour = contours[index]
            p = cv2.arcLength(contour, True)
            s = cv2.contourArea(contour)
            r = cv2.boundingRect(contour)
            if s <= sq_limit and p <= lin_limit:
                out_image[r[1]:r[1]+r[3],r[0]:r[0]+r[2]] = 0
            # choose next contour of the same hierarchy
            index = hierarchy[index][0]

    return out_image


def normalize(img):
    assert len(img.shape) == 2, "Image should have one channel"
    out = img.astype(np.float32)
    mins = out.min()
    maxs = out.max() - mins
    return (out - mins) / (maxs + 0.00001)


def binarize(img, threshold_low=0.0, threshold_high=1.0, size=10, iters=1):
    res = ((img >= threshold_low) & (img <= threshold_high)).astype(np.uint8)
    #res = sieve(res, size)
    #res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=iters)
    #res = cv2.morphologyEx(res, cv2.MORPH_DILATE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return res


def crop_postprocessing(bin_img):
    """
    Mask post-processing for 'Crops'

    - Enlarge pathes and erode boundaries <-> Morpho Erode
    - Smooth forms <-> Smooth countours with median filter
    - No small fields <-> Remove small detections with sieve, linear size < 100 pixels
    - No small holes, do not touch pathes <-> Remove small non-detections with sieve, linear size < 50 pixels

    """

    x = cv2.morphologyEx(bin_img, cv2.MORPH_ERODE, kernel=np.ones((3, 3), dtype=np.uint8), iterations=2)

    x = cv2.medianBlur(x, ksize=7)
    x = cv2.medianBlur(x, ksize=7)
    x = (x > 0.5).astype(np.uint8)

    x = sieve(x, 100)
    h, w = x.shape

    inv_x = (x < 0.5).astype(np.uint8)
    inv_x = inv_x[1:h, 1:w]
    inv_x = sieve(inv_x, 75)
    x[1:h, 1:w] = (inv_x < 0.5).astype(np.uint8)

    x[0:5, :] = bin_img[0:5, :]
    x[:, 0:5] = bin_img[:, 0:5]
    x[-6:, :] = bin_img[-6:, :]
    x[:, -6:] = bin_img[:, -6:]
    return x


# def trees_postprocessing(bin_img):
#     """
#     Mask post-processing for 'Trees'

#     - Smooth forms <-> Smooth countours with median filter
#     - No small holes <-> Remove small non-detections with sieve, linear size < 10 pixels

#     """
#     bin_img = cv2.medianBlur(bin_img, ksize=3)
#     bin_img = (bin_img > 0.55).astype(np.uint8)

#     bin_img = (bin_img < 0.5).astype(np.uint8)
#     h, w = bin_img.shape
#     bin_img[1:h-1, 1:w-1] = sieve(bin_img[1:h-1, 1:w-1], 10)
#     bin_img = (bin_img < 0.5).astype(np.uint8)

#     return bin_img


def path_postprocessing(bin_img):
    """
    Mask post-processing for 'Path' (label 4)

    - Enlarge pathes <-> Morpho dilate + close
    - Smooth forms <-> Smooth countours with median filter
    """
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel=np.ones((3, 3), dtype=np.uint8), iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), dtype=np.uint8), iterations=2)

    bin_img = cv2.medianBlur(bin_img, ksize=3)
    bin_img = (bin_img > 0.55).astype(np.uint8)

    # lines = cv2.HoughLinesP()
    return bin_img


def trees_postprocessing(bin_img):
    """
    Mask post-processing for 'Trees' (label 5)

    - Enlarge trees <-> Morpho dilate
    """
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel=np.ones((3, 3), dtype=np.uint8), iterations=1)
    return bin_img


def buildings_postprocessing(bin_img):
    pass


def mask_postprocessing(labels_image, class_pp_func_list):
    out = np.zeros_like(labels_image)
    for i, l in enumerate(LABELS):
        if i in class_pp_func_list:
            out[:,:,i] = class_pp_func_list[i](labels_image[:,:,i])
        else:
            out[:,:,i] = labels_image[:,:,i]
    return out
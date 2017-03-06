
import os
import sys
sys.path.append("../common/")

import numpy as np
from data_utils import get_filename
from image_utils import generate_pansharpened, generated_upsampled_swir, imwrite
from image_utils import get_image_data, generate_aligned_swir, spot_cleaning
from otb_preprocessing import generate_rm_indices


def create_pan(image_id, sc_threshold=0.1):
    """
    Method to return panchromatic image with cleaned bright spots
    :param image_id:
    :param sc_threshold:
    :return:
    """
    # Open and clean panchromatic image
    img_pan = get_image_data(image_id, 'pan')
    img_pan = spot_cleaning(img_pan, 7, threshold=sc_threshold)
    return img_pan


def create_ms(image_id, sc_threshold=0.1):
    """
    Method to create pansharpened MS + clean bright spots
    :param image_id:
    :param sc_threshold:
    :return:
    """
    # Generate pansharpened MS and upsampled aligned swir
    generate_pansharpened(image_id, 'ms')
    # Open and clean MS image
    img_ms = get_image_data(image_id, 'ms_pan')
    img_ms = spot_cleaning(img_ms, 7, threshold=sc_threshold)
    return img_ms


def create_swir(image_id, sc_threshold=0.1):
    """
    Method to create upsampled SWIR image + clean bright spots
    :param image_id:
    :param sc_threshold:
    :return:
    """
    # Generate aligned swir file
    generate_aligned_swir(image_id)
    # Upsample aligned swir
    generated_upsampled_swir(image_id, 'swir_aligned')
    # Open and clean SWIR image
    img_swir = get_image_data(image_id, 'swir_aligned_upsampled')
    img_swir = spot_cleaning(img_swir, 3, threshold=sc_threshold)
    return img_swir


def create_pan_ms_swir(image_id, sc_threshold=0.1, remove_generated_files=False):
    """
    Method to create an input image with 17 bands: pan, MS, SWIR
    :param image_id:
    :param sc_threshold:
    :param remove_generated_files:
    :return: data ndarray
    """
    img_pan = create_pan(image_id, sc_threshold)
    img_ms = create_ms(image_id, sc_threshold)
    img_swir = create_swir(image_id, sc_threshold)
    h, w = img_pan.shape
    x = np.zeros((h, w, 17), dtype=np.uint16)
    x[:, :, 0] = img_pan
    x[:, :, 1:9] = img_ms
    x[:, :, 9:] = img_swir

    if remove_generated_files:
        os.remove(get_filename(image_id, 'ms_pan'))
        os.remove(get_filename(image_id, 'swir_aligned'))
        os.remove(get_filename(image_id, 'swir_aligned_upsampled'))
    return x


def create_pan_rad_inds_ms(image_id, sc_threshold=0.1, remove_generated_files=False):
    """
    Method to create an input image with 17 channels :
    Panchromatic, NDVI, GEMI, NDWI2, NDTI, BI, BI2, -BI, -BI2, MS
    """

    outfname = get_filename(image_id, '17b')
    if not os.path.exists(outfname):
        img_17b = create_pan_ms_swir(image_id, sc_threshold, remove_generated_files)
        imwrite(outfname, img_17b)
    else:
        img_17b = get_image_data(image_id, '17b')

    generate_rm_indices(image_id)
    img_pan = img_17b[:, :, 0]
    img_ms = img_17b[:, :, 1:9]
    img_multi = get_image_data(image_id, 'multi')
    h, w, nc = img_multi.shape
    # Add inverses of bi, bi2
    # Add ms
    nc2 = nc + 2 + (img_ms.shape[2] + 1)
    # Copy data
    x = np.zeros((h, w, nc2), dtype=np.float32)
    x[:, :, 0] = img_pan.astype(np.float32)
    x[:, :, 1:1 + nc] = img_multi

    def _inverse(img_1b):
        img_1b_max = img_1b.max()
        img_1b_min = img_1b.min()
        return img_1b_max + img_1b_min - img_1b

    x[:, :, 1 + nc] = _inverse(img_multi[:, :, -2])
    x[:, :, 1 + nc + 1] = _inverse(img_multi[:, :, -1])
    x[:, :, 1 + nc + 2:1 + nc + 11] = img_ms

    if remove_generated_files:
        os.remove(get_filename(image_id, 'multi'))
        os.remove(get_filename(image_id, '17b'))

    return x


def _relu(x):
    x = x.copy()
    x[x < 0] = 0
    return x


def water_index_ms(image_id, image_type_ms='ms'):
    img_ms = get_image_data(image_id, image_type_ms).astype(np.float32)
    # image with water as positive values
    a = img_ms[:,:,0].astype(np.float32) - img_ms[:,:,7].astype(np.float32)
    b = img_ms[:,:,1].astype(np.float32) - img_ms[:,:,6].astype(np.float32)
    # enchance positive values with _relu non-linearity
    c = np.sqrt(_relu(a) * _relu(b)) /( a + b + 0.0001)
    return c - np.mean(c)


def water_index_2_ms(image_id, image_type_ms='ms'):
    img_ms = get_image_data(image_id, image_type_ms).astype(np.float32)
    # enchance black pixels from NIR1, NIR2 -> water/buildings are black
    a = np.sqrt(img_ms[:,:,7].astype(np.float32) * img_ms[:,:,6].astype(np.float32))
    # divide by the red band -> buildings are less black
    b = a / (img_ms[:,:,5].astype(np.float32) + 0.0001)
    # inverse colors
    c = 1.0 / (b + 0.0001)
    return c - np.mean(c)


def _CCCI_index(img_ms, img_rgb):
    RE = img_ms[:,:,5]#.astype(np.float32)
    MIR = img_ms[:,:,7]#.astype(np.float32)
    R = img_rgb[:,:,0]#.astype(np.float32)
    # canopy chloropyll content index
    CCCI = (MIR-RE)/(MIR+RE)*(MIR-R)/(MIR+R)
    return CCCI  


def _CCCI_index2(img_ms, img_rgb):
    RE  = img_ms[:,:,5].astype(np.float32)
    MIR = img_ms[:,:,7].astype(np.float32)
    R = img_rgb[:,:,0].astype(np.float32)
    # canopy chloropyll content index
    CCCI = (MIR-RE)/(MIR+RE)*(MIR-R)/(MIR+R)
    return CCCI  


def CCCI_index(image_id, good=False):
    img_rgb = get_image_data(image_id, '3b')
    img_ms = get_image_data(image_id, 'ms_pan')
    h = min(img_rgb.shape[0], img_ms.shape[0])
    w = min(img_rgb.shape[1], img_ms.shape[1])
    if good:
        return _CCCI_index2(img_ms[:h, :w, :], img_rgb[:h, :w, :])
    return _CCCI_index(img_ms[:h, :w, :], img_rgb[:h, :w, :])


def CCCI_index_ms(image_id, image_type_ms='ms'):
    img_ms = get_image_data(image_id, image_type_ms)
    h = img_ms.shape[0]
    w = img_ms.shape[1]
    RE  = img_ms[:,:,5]
    MIR = img_ms[:,:,7]
    R = img_ms[:,:,4]
    return (MIR-RE)/(MIR+RE)*(MIR-R)/(MIR+R)

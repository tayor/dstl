import os
import logging

import numpy as np
import cv2

# GDAL
import gdal
import gdalconst
from gdal_pansharpen import gdal_pansharpen

from data_utils import get_filename, get_resized_polygons
from data_utils import TRAIN_DATA, ORDERED_LABEL_IDS, LABELS


def generate_aligned_swir(image_id):
    """
    Method to create a swir aligned image file
    :param image_id:
    :return:
    """
    outfname = get_filename(image_id, 'swir_aligned')
    if os.path.exists(outfname):
        logging.warn("File '%s' is already existing" % outfname)
        return

    img_pan = get_image_data(image_id, 'pan')
    img_swir = get_image_data(image_id, 'swir')

    img_swir_aligned = compute_aligned_image(img_pan, img_swir)
    imwrite(outfname, img_swir_aligned)


def generated_upsampled_swir(image_id, image_type):
    """
    Method to generate an upsampled swir image
    :param image_id:
    :param image_type: 'swir' or 'swir*'
    """
    assert 'swir' in image_type, "Image type should be derived from 'swir'"
    outfname = get_filename(image_id, image_type + '_upsampled')
    if os.path.exists(outfname):
        logging.warn("File '%s' is already existing" % outfname)
        return

    h, w, _ = get_image_data(image_id, 'pan', return_shape_only=True)
    img_swir = get_image_data(image_id, image_type)
    img_swir_upsampled = cv2.resize(img_swir, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
    imwrite(outfname, img_swir_upsampled)


def generate_pansharpened(image_id, image_type):
    """
    Method to create pansharpened images from multispectral or swir images
    Created file is placed in GENERATED_DATA folder

    :image_type: 'ms' or 'ms*'
    """
    assert 'ms' in image_type, "Image type should be derived from 'ms'"
    outfname = get_filename(image_id, image_type + '_pan')
    if os.path.exists(outfname):
        logging.warn("File '%s' is already existing" % outfname)
        return

    fname = get_filename(image_id, image_type)
    fname_pan = get_filename(image_id, 'pan')
    gdal_pansharpen(['', fname_pan, fname, outfname])


def print_image_info(image_id, image_type):

    fname = get_filename(image_id, image_type)
    img = gdal.Open(fname, gdalconst.GA_ReadOnly)
    assert img, "Image file is not found: {}".format(fname)

    print("Image size:", img.RasterYSize, img.RasterXSize, img.RasterCount)
    print("Metadata:", img.GetMetadata_List())
    print("MetadataDomainList:", img.GetMetadataDomainList())
    print("Description:", img.GetDescription())
    print("ProjectionRef:", img.GetProjectionRef())
    print("GeoTransform:", img.GetGeoTransform())


def get_image_tile_data(fname, return_shape_only=False):
    """
    Method to get image tile data as np.array
    """
    img = gdal.Open(fname, gdalconst.GA_ReadOnly)
    assert img, "Image file is not found: {}".format(fname)
    if return_shape_only:
        return img.RasterYSize, img.RasterXSize, img.RasterCount

    img_data = img.ReadAsArray()
    if len(img_data.shape) == 3:
        return img_data.transpose([1, 2, 0])
    return img_data


def get_image_data(image_id, image_type, return_shape_only=False):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = gdal.Open(fname, gdalconst.GA_ReadOnly)
    assert img, "Image is not found: id={}, type={}".format(image_id, image_type)
    if return_shape_only:
        return (img.RasterYSize, img.RasterXSize, img.RasterCount)

    img_data = img.ReadAsArray()
    if len(img_data.shape) == 3:
        return img_data.transpose([1, 2, 0])
    return img_data


def imwrite(filename, data, compress=True):
    driver = gdal.GetDriverByName("GTiff")
    data_type = to_gdal(data.dtype)
    nb_bands = data.shape[2]
    width = data.shape[1]
    height = data.shape[0]

    kwargs = {}
    if compress:
        kwargs['options'] = ['COMPRESS=LZW']
    dst_dataset = driver.Create(filename, width, height, nb_bands, data_type, **kwargs)
    assert dst_dataset is not None, "File '%s' is not created" % filename
    for band_index in range(1,nb_bands+1):
        dst_band = dst_dataset.GetRasterBand(band_index)
        dst_band.WriteArray(data[:,:,band_index-1], 0, 0)


def to_gdal(dtype):
    """ Method to convert numpy data type to Gdal data type """
    if dtype == np.uint8:
        return gdal.GDT_Byte
    elif dtype == np.int16:
        return gdal.GDT_Int16
    elif dtype == np.int32:
        return gdal.GDT_Int32
    elif dtype == np.uint16:
        return gdal.GDT_UInt16
    elif dtype == np.uint32:
        return gdal.GDT_UInt32
    elif dtype == np.float32:
        return gdal.GDT_Float32
    elif dtype == np.float64:
        return gdal.GDT_Float64
    elif dtype == np.complex64:
        return gdal.GDT_CFloat32
    elif dtype == np.complex128:
        return gdal.GDT_CFloat64
    else:
        return gdal.GDT_Unknown


def normalize(in_img, q_min=0.5, q_max=99.5, return_mins_maxs=False):
    """
    Normalize image in [0.0, 1.0]
    mins is array of minima
    maxs is array of differences between maxima and minima
    """
    init_shape = in_img.shape
    if len(init_shape) == 2:
        in_img = np.expand_dims(in_img, axis=2)
    w, h, d = in_img.shape
    img = in_img.copy()
    img = np.reshape(img, [w * h, d]).astype(np.float64)
    mins = np.percentile(img, q_min, axis=0)
    maxs = np.percentile(img, q_max, axis=0) - mins
    maxs[(maxs < 0.0001) & (maxs > -0.0001)] = 0.0001
    img = (img - mins[None, :]) / maxs[None, :]
    img = img.clip(0.0, 1.0)
    img = np.reshape(img, [w, h, d])
    if init_shape != img.shape:
        img = img.reshape(init_shape)
    if return_mins_maxs:
        return img, mins, maxs
    return img


def get_gradient(im):
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)

    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad


def compute_alignment_warp_matrix(img_master, img_slave, roi, warp_mode=cv2.MOTION_TRANSLATION):
    """
    Code taken from http://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    """
    fx = img_master.shape[1] * 1.0 / img_slave.shape[1]
    fy = img_master.shape[0] * 1.0 / img_slave.shape[0]
    roi_slave = [int(roi[0] / fx), int(roi[1] / fy), int(roi[2] / fx), int(roi[3] / fy)]

    img_slave_roi = img_slave[roi_slave[1]:roi_slave[3], roi_slave[0]:roi_slave[2], :].astype(np.float32)

    img_master_roi = img_master[roi[1]:roi[3], roi[0]:roi[2]].astype(np.float32)
    img_master_roi = cv2.resize(img_master_roi, dsize=(img_slave_roi.shape[1], img_slave_roi.shape[0]))

    img_master_roi = get_gradient(img_master_roi)
    img_slave_roi = get_gradient(img_slave_roi)

    height, width, ll = img_slave.shape

    # Set the warp matrix to identity.
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        mean_warp_matrix = np.zeros((3, 3), dtype=np.float32)
    else:
        mean_warp_matrix = np.zeros((2, 3), dtype=np.float32)

    for i in range(ll):

        # Set the warp matrix to identity.
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        # Set the stopping criteria for the algorithm.
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 0.01)
        try:
            cc, warp_matrix = cv2.findTransformECC(img_master_roi,
                                                   img_slave_roi[:, :, i],
                                                   warp_matrix,
                                                   warp_mode,
                                                   criteria)
        except Exception as e:
            logging.error("Failed to find warp matrix: %s" % str(e))
            return warp_matrix

        mean_warp_matrix += warp_matrix

    mean_warp_matrix *= 1.0/ll
    return mean_warp_matrix


def compute_aligned_image(img_master, img_slave):
    # Compute mean warp matrix
    roi=[0,0,500,500]
    warp_mode = cv2.MOTION_TRANSLATION
    mean_warp_matrix = np.zeros((2, 3), dtype=np.float32)
    mean_warp_matrix[0, 0] = 1.0
    mean_warp_matrix[1, 1] = 1.0
    tx = []
    ty = []
    n = 3
    for i in range(n):
        for j in range(n):
            warp_matrix = compute_alignment_warp_matrix(img_master, img_slave, roi=roi, warp_mode=warp_mode)
            tx.append(warp_matrix[0, 2])
            ty.append(warp_matrix[1, 2])
            roi[0] = i * 500
            roi[1] = j * 500
            roi[2] += roi[0]
            roi[3] += roi[1]

    tx = np.median(tx)
    ty = np.median(ty)
    mean_warp_matrix[0, 2] = tx
    mean_warp_matrix[1, 2] = ty

    #print "mean_warp_matrix :"
    #print mean_warp_matrix

    img_slave_aligned = np.zeros_like(img_slave)
    height, width, ll = img_slave.shape
    for i in range(ll):
        # Use Affine warp when the transformation is not a Homography
        img_slave_aligned[:, :, i] = cv2.warpAffine(img_slave[:, :, i],
                                                    mean_warp_matrix,
                                                    (width, height),
                                                    flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                                                    borderMode=cv2.BORDER_REPLICATE
                                                    )

    img_slave_aligned = img_slave if img_slave_aligned is None else img_slave_aligned
    return img_slave_aligned


def median_blur(img, ksize):
    init_shape = img.shape
    img2 = np.expand_dims(img, axis=2) if len(init_shape) == 2 else img
    out = np.zeros_like(img2)
    img_n, mins, maxs = normalize(img2, return_mins_maxs=True)
    for i in range(img2.shape[2]):
        img_temp = (255.0*img_n[:,:,i]).astype(np.uint8)
        img_temp = 1.0/255.0 * cv2.medianBlur(img_temp, ksize).astype(img.dtype)
        out[:,:,i] = maxs[i] * img_temp + mins[i]
    out = out.reshape(init_shape)
    return out


def spot_cleaning(img, ksize, threshold=0.15):
    """
    ksize : kernel size for median blur
    threshold for outliers, [0.0,1.0]
    https://github.com/kmader/Quantitative-Big-Imaging-2016/blob/master/Lectures/02-Slides.pdf
    """
    init_type = img.dtype
    init_shape = img.shape
    if len(init_shape) == 2:
        img = img[:, :, None]
    img_median = median_blur(img, ksize).astype(np.float32)
    diff = np.abs(img.astype(np.float32) - img_median)
    diff = np.mean(diff, axis=2)
    diff = normalize(diff, q_min=0, q_max=100)
    diff2 = diff.copy()
    _, diff = cv2.threshold(diff, threshold, 1.0, cv2.THRESH_BINARY)
    _, diff2 = cv2.threshold(diff2, threshold, 1.0, cv2.THRESH_BINARY_INV)

    img_median2 = img_median * diff[:, :, None]
    img2 = img * diff2[:, :, None]
    img2 += img_median2
    if img2.shape != init_shape:
        img2 = img2.reshape(init_shape)
    return img2.astype(init_type)


def align_images(img_master, img_slave, roi, warp_mode=cv2.MOTION_TRANSLATION):
    """
    Code taken from http://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    """
    fx = img_master.shape[1] * 1.0 / img_slave.shape[1]
    fy = img_master.shape[0] * 1.0 / img_slave.shape[0]
    roi_slave = [int(roi[0] / fx), int(roi[1] / fy), int(roi[2] / fx), int(roi[3] / fy)]

    img_slave_roi = img_slave[roi_slave[1]:roi_slave[3], roi_slave[0]:roi_slave[2], :].astype(np.float32)

    img_master_roi = img_master[roi[1]:roi[3], roi[0]:roi[2]].astype(np.float32)
    img_master_roi = cv2.resize(img_master_roi, dsize=(img_slave_roi.shape[1], img_slave_roi.shape[0]))

    img_master_roi = get_gradient(img_master_roi)
    img_slave_roi = get_gradient(img_slave_roi)

    img_slave_aligned = np.zeros_like(img_slave)
    height, width, ll = img_slave.shape
    # Set the warp matrix to identity.
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        mean_warp_matrix = np.zeros((3, 3), dtype=np.float32)
    else:
        mean_warp_matrix = np.zeros((2, 3), dtype=np.float32)

    for i in range(ll):

        # Set the warp matrix to identity.
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        # Set the stopping criteria for the algorithm.
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 0.01)
        try:
            cc, warp_matrix = cv2.findTransformECC(img_master_roi,
                                                   img_slave_roi[:, :, i],
                                                   warp_matrix,
                                                   warp_mode,
                                                   criteria)
        except Exception as e:
            logging.error("Failed to find warp matrix: %s" % str(e))
            return None

        mean_warp_matrix += warp_matrix

    mean_warp_matrix *= 1.0/ll

    for i in range(ll):

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use Perspective warp when the transformation is a Homography
            img_slave_aligned[:, :, i] = cv2.warpPerspective(img_slave[:, :, i],
                                                             mean_warp_matrix,
                                                             (width, height),
                                                             flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                                                             borderMode=cv2.BORDER_REPLICATE
                                                             )
        else:
            # Use Affine warp when the transformation is not a Homography
            img_slave_aligned[:, :, i] = cv2.warpAffine(img_slave[:, :, i],
                                                        mean_warp_matrix,
                                                        (width, height),
                                                        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                                                        borderMode=cv2.BORDER_REPLICATE
                                                        )

    return img_slave_aligned


def make_ratios_vegetation(img_17b):
    """
        Method creates an image of all possible band ratios

        - panchromatic[0] / MS[5] = Trees, Crops, Misc manmade structures (of trees)
        - panchromatic[0] / MS[4] = Trees, Crops, Misc manmade structures (of trees)
        - MS[1] / MS[5] = Trees, Crops, Misc manmade structures (of trees)
        - MS[2] / MS[5] = Trees, Crops, Misc manmade structures (of trees)
        - MS[6] / MS[4] = Trees, Crops, Misc manmade structures (of trees)
        - MS[6] / MS[5] = Trees, Crops, Misc manmade structures (of trees)
        - MS[7] / MS[4] = Trees, Crops, Misc manmade structures (of trees)
        - MS[7] / MS[5] = Trees, Crops, Misc manmade structures (of trees)
        - MS[7] / MS[10:17] = Trees, Crops, Misc manmade structures (of trees)
        - MS[8] / MS[4] = Trees, Crops, Misc manmade structures (of trees)
        - MS[8:17] / MS[5] = Trees, Crops, Misc manmade structures (of trees)
    """
    h, w, n = img_17b.shape

    out_n = 23
    out = np.zeros((h, w, out_n), dtype=np.float32)
    def _ratio(i, j):
        return img_17b[:,:,i] / (img_17b[:,:,j] + 0.00001)

    c = 0
    out[:,:,c] = _ratio(0, 5); c+= 1
    out[:,:,c] = _ratio(0, 4); c+= 1
    out[:,:,c] = _ratio(1, 5); c+= 1
    out[:,:,c] = _ratio(2, 5); c+= 1
    out[:,:,c] = _ratio(6, 4); c+= 1
    out[:,:,c] = _ratio(6, 5); c+= 1
    out[:,:,c] = _ratio(7, 5); c+= 1
    out[:,:,c] = _ratio(7, 4); c+= 1
    out[:,:,c] = _ratio(8, 4); c+= 1
    out[:,:,c] = _ratio(7, 10); c+= 1
    out[:,:,c] = _ratio(7, 11); c+= 1
    out[:,:,c] = _ratio(7, 12); c+= 1
    out[:,:,c] = _ratio(7, 13); c+= 1
    out[:,:,c] = _ratio(7, 14); c+= 1
    out[:,:,c] = _ratio(7, 15); c+= 1
    out[:,:,c] = _ratio(7, 16); c+= 1
    out[:,:,c] = _ratio(9, 5); c+= 1
    out[:,:,c] = _ratio(10, 5); c+= 1
    out[:,:,c] = _ratio(11, 5); c+= 1
    out[:,:,c] = _ratio(12, 5); c+= 1
    out[:,:,c] = _ratio(13, 5); c+= 1
    out[:,:,c] = _ratio(14, 5); c+= 1
    out[:,:,c] = _ratio(15, 5); c+= 1
    return out


def compute_mean_std_on_tiles(trainset_ids):
    """
    Method to compute mean/std tile image
    :return: mean_tile_image, std_tile_image
    """
    ll = len(trainset_ids)
    tile_id = trainset_ids[0]
    mean_tile_image = get_image_tile_data(os.path.join(TRAIN_DATA,tile_id)).astype(np.float32)
    # Init mean/std images
    std_tile_image = np.power(mean_tile_image, 2)

    for i, tile_id in enumerate(trainset_ids[1:]):
        logging.info("-- %i/%i | %s" % (i+2, ll, tile_id))
        tile = get_image_tile_data(os.path.join(TRAIN_DATA,tile_id)).astype(np.float32)
        mean_tile_image += tile
        std_tile_image += np.power(tile, 2)

    mean_tile_image *= 1.0/ll
    std_tile_image *= 1.0/ll
    std_tile_image -= np.power(mean_tile_image, 2)
    std_tile_image = np.sqrt(std_tile_image)
    return mean_tile_image, std_tile_image


def compute_mean_std_on_images(trainset_ids, image_type='input', feature_wise=False, out_shape=None):
    """
    Method to compute mean/std input image
    :return: mean_image, std_image
    """

    max_dims = [0, 0]
    nc = 0
    if out_shape is None:
        for image_id in trainset_ids:
            shape = get_image_data(image_id, image_type, return_shape_only=True)
            if shape[0] > max_dims[0]:
                max_dims[0] = shape[0]
            if shape[1] > max_dims[1]:
                max_dims[1] = shape[1]
            if shape[2] > nc:
                nc = shape[2]
    else:
        max_dims = out_shape[:2]
        nc = out_shape[2]

    ll = len(trainset_ids)
    # Init mean/std images
    mean_image = np.zeros(tuple(max_dims) + (nc, ), dtype=np.float32)
    std_image = np.zeros(tuple(max_dims) + (nc,), dtype=np.float32)
    for i, image_id in enumerate(trainset_ids):
        logging.info("-- %i/%i | %s" % (i + 1, ll, image_id))
        img_Kb = get_image_data(image_id, image_type).astype(np.float32)
        h, w, _ = img_Kb.shape
        if feature_wise:
            mean_image[:, :, :] += np.mean(img_Kb, axis=(0, 1))
            std_image[:, :, :] += np.std(img_Kb, axis=(0, 1))
        else:
            mean_image[:h, :w, :] += img_Kb
            std_image[:h, :w, :] += np.power(img_Kb, 2.0)

    mean_image *= 1.0 / ll
    std_image *= 1.0 / ll
    if not feature_wise:
        std_image -= np.power(mean_image, 2.0)
        std_image = np.sqrt(std_image)
    return mean_image, std_image


def generate_label_file(image_id, multi_dim=True):
    if multi_dim:
        outfname = get_filename(image_id, 'label')
        if os.path.exists(outfname):
            logging.warn("File '%s' is already existing" % outfname)
            return
        image_data = generate_label_image2(image_id)
        imwrite(outfname, image_data)
    else:
        outfname = get_filename(image_id, 'label_1d')
        if os.path.exists(outfname):
            logging.warn("File '%s' is already existing" % outfname)
            return
        image_data = generate_label_image(image_id)
        cv2.imwrite(outfname, image_data)


def generate_label_image(image_id, image_type='pan', labels=None):

    image_shape = get_image_data(image_id, image_type, return_shape_only=True)
    rpolygons = get_resized_polygons(image_id, *image_shape[:2])
    out = np.zeros(image_shape[:2], np.uint8)
    round_coords = lambda x: np.array(x).round().astype(np.int32)
    label_ids = ORDERED_LABEL_IDS if labels is None else [i for i in ORDERED_LABEL_IDS if i in labels]
    for i, class_type in enumerate(label_ids):
        if class_type not in rpolygons:
            continue
        one_class_mask = np.zeros(out.shape[:2], np.uint8)
        for polygon in rpolygons[class_type]:
            exterior = [round_coords(polygon.exterior.coords)]
            cv2.fillPoly(one_class_mask, exterior, i)
            if len(polygon.interiors) > 0:
                interiors = [round_coords(poly.coords) for poly in polygon.interiors]
                cv2.fillPoly(one_class_mask, interiors, 0)
        out = np.maximum(out, one_class_mask)
    return out


def generate_label_image2(image_id, image_type='pan'):

    image_shape = get_image_data(image_id, image_type, return_shape_only=True)
    rpolygons = get_resized_polygons(image_id, *image_shape[:2])    
    out = np.zeros(image_shape[:2] + (len(LABELS), ), np.uint8)
    out[:,:,0] = 1
    round_coords = lambda x: np.array(x).round().astype(np.int32)
    for class_type in range(1, len(LABELS)):
        if class_type not in rpolygons:
            continue
        one_class_mask = np.zeros(out.shape[:2], np.uint8)
        for polygon in rpolygons[class_type]:
            exterior = [round_coords(polygon.exterior.coords)]
            cv2.fillPoly(one_class_mask, exterior, 1)
            if len(polygon.interiors) > 0:
                interiors = [round_coords(poly.coords) for poly in polygon.interiors]
                cv2.fillPoly(one_class_mask, interiors, 0)
        out[:,:,class_type] = one_class_mask
        out[:,:,0] = np.bitwise_xor(out[:,:,0], np.bitwise_and(out[:,:,0], one_class_mask)) # =x ^ (x & y)
    return out


def get_common_size(img1, img2):
    return min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])

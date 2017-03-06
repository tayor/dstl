import os
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2

from shapely.wkt import loads
from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Polygon


assert os.path.exists('../input'), "Please download Kaggle input data into 'input' folder"

GRID_SIZE = pd.read_csv('../input/grid_sizes.csv',
                        names=['ImageId', 'Xmax', 'Ymin'],
                        skiprows=1)
GRID_SIZE.columns = ['ImageId', 'Xmax', 'Ymin']
TRAIN_WKT = pd.read_csv('../input/train_wkt_v4.csv',
                        dtype={'ClassType':int})

TRAIN_IMAGE_IDS = TRAIN_WKT['ImageId'].unique()
ALL_IMAGE_IDS = GRID_SIZE['ImageId'].unique()

DATA_3_BANDS = os.path.join('..', 'input', 'three_band')
DATA_16_BANDS = os.path.join('..', 'input', 'sixteen_band')
GENERATED_DATA = os.path.join('..', 'input', 'generated')
TRAIN_DATA = os.path.join('..', 'input', 'train')
TEST_DATA = os.path.join('..', 'input', 'test')

GENERATED_LABELS = os.path.join(GENERATED_DATA, 'labels')
TRAIN_LABEL_TILES = os.path.join(TRAIN_DATA, 'labels')
TRAIN_LABEL_TILES_1D = os.path.join(TRAIN_DATA, 'labels_1d')

if not os.path.exists(GENERATED_DATA):
    os.makedirs(GENERATED_DATA)

if not os.path.exists(GENERATED_LABELS):
    os.makedirs(GENERATED_LABELS)

if not os.path.exists(TRAIN_DATA):
    os.makedirs(TRAIN_DATA)

if not os.path.exists(TEST_DATA):
    os.makedirs(TEST_DATA)

LABELS = [
    "None",
    # 1
    "Buildings",
    # 2
    "Misc. Manmade structures",
    # 3
    "Road",
    # 4
    "Track",
    # 5
    "Trees",
    # 6
    "Crops",
    # 7
    "Waterway",
    # 8
    "Standing water",
    # 9
    "Vehicle Large",
    # 10
    "Vehicle Small",
]

ORDERED_LABEL_IDS = [
    0, # Nothing
    6, # Crops
    4, # Track
    3, # Road
    8, # Standing water
    7, # Waterway
    2, # Structures
    10, # Small vehicle
    9, # Large vehicle
    1, # Building
    5, # Trees
]

FULL_LABELS = [
    None,
    # 1
    "Buildings - large building, residential, non-residential, fuel storage facility, fortified building",
    # 2
    "Misc. Manmade structures",
    # 3
    "Road",
    # 4
    "Track - poor/dirt/cart track, footpath/trail",
    # 5
    "Trees - woodland, hedgerows, groups of trees, standalone trees",
    # 6
    "Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops",
    # 7
    "Waterway",
    # 8
    "Standing water",
    # 9
    "Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle",
    # 10
    "Vehicle Small - small vehicle (car, van), motorbike",
]


def get_tile_filename(image_id, xoffset, yoffset, image_type):
    if image_type == '17b':
        dir_path = TRAIN_DATA
        ext = 'tif'
    elif image_type == 'label':
        dir_path = TRAIN_LABEL_TILES
        ext = 'tif'
    elif image_type == 'label_1d':
        dir_path = TRAIN_LABEL_TILES_1D
        ext = 'tif'
    else:
        raise Exception("Unknown image type: {}".format(image_type))
    return os.path.join(dir_path, "{}_{}_{}.{}".format(image_id, xoffset, yoffset, ext))

    
def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type   
    """
    ext = 'tif'
    if image_type == '3b':
        data_path = DATA_3_BANDS
        suffix = ''
    elif image_type == 'pan':
        data_path = DATA_16_BANDS
        suffix = '_P'
    elif image_type == 'ms':
        data_path = DATA_16_BANDS
        suffix = '_M'
    elif image_type == 'swir':
        data_path = DATA_16_BANDS
        suffix = '_A'
    elif image_type == 'swir_aligned':
        data_path = GENERATED_DATA
        suffix = '_A_aligned'
    elif image_type == 'ms_pan':
        data_path = GENERATED_DATA
        suffix = '_M_P'
    elif image_type == 'swir_upsampled' or \
            image_type == 'swir_aligned_upsampled':
        data_path = GENERATED_DATA
        suffix = '_A_P'
    elif image_type == 'label_1d':
        data_path = GENERATED_LABELS
        suffix = '_1d'
    elif image_type == 'label':
        data_path = GENERATED_LABELS
        suffix = ''
    elif image_type == '17b':
        data_path = GENERATED_DATA
        suffix = ''
    elif image_type == 'multi':
        data_path = GENERATED_DATA
        suffix = '_multi'
    elif image_type == 'input':
        data_path = TRAIN_DATA if image_id in TRAIN_IMAGE_IDS else TEST_DATA
        suffix = ''
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)
        
    return os.path.join(data_path, "{}{}.{}".format(image_id, suffix, ext))


def get_unit_polygons(image_id, classes=None):
    polygons = {}
    image_polygons = TRAIN_WKT[TRAIN_WKT['ImageId'] == image_id]
    assert not image_polygons.empty, "This is a test image"
    classes = list(range(1,len(LABELS))) if classes is None else classes
    for class_type in classes:
        polygons[class_type] = loads(image_polygons[image_polygons['ClassType'] == class_type].MultipolygonWKT.values[0])
    return polygons


def get_grid_params(image_id):
    params = GRID_SIZE[GRID_SIZE['ImageId'] == image_id][['Xmax', 'Ymin']]
    assert not params.empty, "No grid parameters for this image id : {}".format(image_id)
    return params.values[0]


def get_scalers(image_id, image_height, image_width):
    x_max, y_min = get_grid_params(image_id)
    h, w = image_height, image_width
    w_ = w * (w * 1.0 / (w + 1.0))
    h_ = h * (h * 1.0 / (h + 1.0))
    return w_ / x_max, h_ / y_min


def get_resized_polygons(image_id, image_height, image_width, classes=None):
    polygons = get_unit_polygons(image_id, classes)
    x_scaler, y_scaler = get_scalers(image_id, image_height, image_width)
    resized_polygons = {}
    for class_type in polygons:
        poly = polygons[class_type]
        rpoly = scale(poly, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))  
        resized_polygons[class_type] = rpoly
    return resized_polygons
    
  
def get_image_ids(classes, gb):
    image_ids = set()
    for c in classes:
        ids = gb.get_group(c)['ImageId'].values.tolist()
        image_ids.update(ids)
    return list(image_ids)


def mask_to_polygons(mask, epsilon=5, min_area=1.):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    if epsilon > 0:
        approx_contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]
    else:
        approx_contours = contours

    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons

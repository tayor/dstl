
#
# Methods to work with submissions
#

import numpy as np
import cv2

from shapely.wkt import loads
from shapely.affinity import scale
from shapely.geometry import mapping
from shapely.wkt import dumps
import fiona


import sys
sys.path.append("../common/")
from data_utils import get_scalers, LABELS
from image_utils import get_image_data
from data_utils import mask_to_polygons


def _process(line):
    data = line.split(',', 2)
    data[2] = data[2].replace("\"", "")
    data[2] = data[2].replace("\n", "")
    data[2] = data[2].replace("\r", "")
    return data


def _unprocess(poly_info):
    return ",".join([poly_info[0], str(poly_info[1]), "\"" + poly_info[2] + "\""]) + "\r\n"


def submission_iterator(csv_file):
    f_in = open(csv_file)
    _ = f_in.readline()

    data_csv = []
    next_data_csv = []

    while True:

        # First line
        if len(next_data_csv) == 0:
            line = f_in.readline()
            # print "First line: ", line[:35], len(data_csv), len(next_data_csv)
            if len(line) == 0:
                return
            image_id = line[:8]
            data_csv.append(_process(line))
        else:
            data_csv = []
            data_csv.append(next_data_csv)
            # print "Copy next_data_csv -> data_csv", next_data_csv[:2], data_csv[0][:2]
            image_id = next_data_csv[0]
            next_data_csv = []

        # Loop
        counter = 0
        while counter < 15:
            prev_image_id = image_id
            line = f_in.readline()
            if len(line) == 0:
                if len(data_csv) > 0:
                    yield data_csv
                return
            image_id = line[:8]
            if image_id == prev_image_id:
                data_csv.append(_process(line))
            else:
                next_data_csv = _process(line)
                # print "End of unique ImageId : ", len(data_csv), "Next is", next_data_csv[0], next_data_csv[1]
                yield data_csv
                break


def write_shp_from_csv(filename, data_csv, simplify=False, tol=5, n_pts=15):
    # Write a new Shapefile
    image_id = data_csv[0][0]
    h, w, _ = get_image_data(image_id, '3b', return_shape_only=True)
    xs, ys = get_scalers(image_id, h, w)

    all_scaled_polygons = {}
    for poly_info in data_csv:
        if "MULTIPOLYGON" not in poly_info[2][:20]:
            continue
        polygons = loads(poly_info[2])
        scaled_polygons = scale(polygons, xfact=xs, yfact=ys, origin=(0, 0, 0))
        all_scaled_polygons[int(poly_info[1])] = scaled_polygons
        if simplify:
            for k in scaled_polygons:
                if len(scaled_polygons[k].exterior.coords) > n_pts:
                    scaled_polygons[k] = scaled_polygons[k].simplify(tolerance=5)
    write_shp_from_polygons(filename, image_id, all_scaled_polygons)


def write_shp_from_polygons(filename, image_id, all_scaled_polygons):
    schema = {
        'geometry': 'MultiPolygon',
        'properties': {'image_id': 'str', 'class': 'int'},
    }
    with fiona.open(filename, 'w', 'ESRI Shapefile', schema) as c:
        for k in all_scaled_polygons:
            polygons = all_scaled_polygons[k]
            # To align with images in QGis
            polygons = scale(polygons, xfact=1.0, yfact=-1.0, origin=(0, 0, 0))
            c.write({
                'geometry': mapping(polygons),
                'properties': {'image_id': image_id, 'class': k},
            })
    print "Written succesfully file : ", filename


def get_data_csv(image_id, csv_file):
    for data_csv in submission_iterator(csv_file):
        if image_id == data_csv[0][0]:
            return data_csv


def compute_label_image(image_id, scaled_polygons, image_type='3b'):
    image_shape = get_image_data(image_id, image_type, return_shape_only=True)
    out = np.zeros((image_shape[0], image_shape[1], len(LABELS)), np.uint8)
    out[:, :, 0] = 1
    round_coords = lambda x: np.array(x).round().astype(np.int32)
    for class_type in range(1, len(LABELS)):
        if class_type not in scaled_polygons:
            continue
        one_class_mask = np.zeros((image_shape[0], image_shape[1]), np.uint8)
        for polygon in scaled_polygons[class_type]:
            exterior = [round_coords(polygon.exterior.coords)]
            cv2.fillPoly(one_class_mask, exterior, 1)
            if len(polygon.interiors) > 0:
                interiors = [round_coords(poly.coords) for poly in polygon.interiors]
                cv2.fillPoly(one_class_mask, interiors, 0)
        out[:, :, class_type] = one_class_mask
        out[:, :, 0] = np.bitwise_xor(out[:, :, 0], np.bitwise_and(out[:, :, 0], one_class_mask))  # =x ^ (x & y)
    return out


def get_polygons(data_csv):
    out = {}
    for poly_info in data_csv:
        if "MULTIPOLYGON" not in poly_info[2][:20]:
            continue
        polygons = loads(poly_info[2])
        out[poly_info[1]] = polygons
    return out


def get_scaled_polygons(data_csv, image_type='3b'):
    out = {}
    image_id = data_csv[0][0]
    h, w, _ = get_image_data(image_id, image_type, return_shape_only=True)
    xs, ys = get_scalers(image_id, h, w)
    for poly_info in data_csv:
        if "MULTIPOLYGON" not in poly_info[2][:20]:
            continue
        polygons = loads(poly_info[2])
        scaled_polygons = scale(polygons, xfact=xs, yfact=ys, origin=(0, 0, 0))
        out[int(poly_info[1])] = scaled_polygons
    return out


def write_shp_from_mask(filename, image_id, labels_image, epsilon=0, min_area=0.1):
    all_scaled_polygons = {}
    for class_index in range(1, len(LABELS)):
        polygons = mask_to_polygons(labels_image[:, :, class_index], epsilon=epsilon, min_area=min_area)
        all_scaled_polygons[class_index] = polygons
    write_shp_from_polygons(filename, image_id, all_scaled_polygons)


def rewrite_submission(input_csv_filename, output_csv_file, postproc_single_class_functions):
    """

    :param input_csv_filename:
    :param output_csv_file:
    :param postproc_single_class_functions: { i: pp_class_1_func }
    :return:
    """
    empty_polygon = 'MULTIPOLYGON EMPTY'

    data_iterator = submission_iterator(input_csv_filename)

    f_out = open(output_csv_file, 'w')

    f_out.write("ImageId,ClassType,MultipolygonWKT\r\n")
    try:
        index = 0
        round_coords = lambda x: np.array(x).round().astype(np.int32)
        for data_csv in data_iterator:
            print "--", data_csv[0][0], len(data_csv), index
            image_id = data_csv[0][0]

            h, w, _ = get_image_data(image_id, '3b', return_shape_only=True)
            x_scaler, y_scaler = get_scalers(image_id, h, w)
            for i, class_index in enumerate(range(1, len(LABELS))):
                if "EMPTY" in data_csv[i][2][:50]:
                    f_out.write(_unprocess(data_csv[i]))
                    continue

                if class_index in postproc_single_class_functions:
                    polygons = loads(data_csv[i][2])
                    scaled_polygons = scale(polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
                    one_class_mask = np.zeros((h, w), np.uint8)
                    for polygon in scaled_polygons:
                        exterior = [round_coords(polygon.exterior.coords)]
                        cv2.fillPoly(one_class_mask, exterior, 1)
                        if len(polygon.interiors) > 0:
                            interiors = [round_coords(poly.coords) for poly in polygon.interiors]
                            cv2.fillPoly(one_class_mask, interiors, 0)
                    pp_one_class_mask = postproc_single_class_functions[class_index](one_class_mask)
                    polygons = mask_to_polygons(pp_one_class_mask, epsilon=0.0, min_area=0.1)
                    if len(polygons) == 0:
                        line = ",".join([image_id, str(class_index), empty_polygon]) + "\r\n"
                        f_out.write(line)
                    else:
                        unit_polygons = scale(polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler, origin=(0, 0, 0))
                        line = ",".join([image_id, str(class_index), "\"" + dumps(unit_polygons) + "\""]) + "\r\n"
                        f_out.write(line)
                else:
                    f_out.write(_unprocess(data_csv[i]))
            index += 1
    except KeyboardInterrupt:
        pass

    f_out.close()


def rewrite_submission2(input_csv_filename, output_csv_file, postproc_function):
    empty_polygon = 'MULTIPOLYGON EMPTY'

    data_iterator = submission_iterator(input_csv_filename)

    f_out = open(output_csv_file, 'w')

    f_out.write('ImageId,ClassType,MultipolygonWKT\r\n')
    try:
        index = 0
        for data_csv in data_iterator:
            print "--", data_csv[0][0], len(data_csv), index
            image_id = data_csv[0][0]

            polygons = get_scaled_polygons(data_csv)
            labels_image = compute_label_image(image_id, polygons)
            pp_labels_image = postproc_function(labels_image)

            x_scaler, y_scaler = get_scalers(image_id, pp_labels_image.shape[0], pp_labels_image.shape[1])
            for class_index in range(1, len(LABELS)):

                polygons = mask_to_polygons(pp_labels_image[:, :, class_index - 1], epsilon=0.0, min_area=0.1)
                if len(polygons) == 0:
                    line = ",".join([image_id, str(class_index), empty_polygon]) + "\r\n"
                    f_out.write(line)
                else:
                    unit_polygons = scale(polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler, origin=(0, 0, 0))
                    line = ",".join([image_id, str(class_index), "\"" + dumps(unit_polygons) + "\""]) + "\r\n"
                    f_out.write(line)
            index += 1

    except KeyboardInterrupt:
        pass

    f_out.close()
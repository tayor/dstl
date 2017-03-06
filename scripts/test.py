
#
# Script to compute predictions using a trained model on all test data
#
import os
import logging
from datetime import datetime
import sys
sys.path.append("../common/")

import numpy as np
import pandas as pd

from shapely.affinity import scale
from shapely.wkt import dumps

from data_utils import ALL_IMAGE_IDS, TRAIN_IMAGE_IDS, LABELS, TRAIN_DATA
from data_utils import get_filename, mask_to_polygons, get_scalers
from image_utils import get_image_data, imwrite
from geo_utils.GeoImage import GeoImage
from predictions_utils import compute_predictions

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def load_model(model_json_filename, weights_filename):
    from keras.models import model_from_json
    # load json and create model
    json_file = open(model_json_filename, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights(weights_filename)
    logger.info("Loaded model from files: %s, %s" % (model_json_filename, weights_filename))
    return model


def create_submission():
    empty_polygon = 'MULTIPOLYGON EMPTY'
    ll = len(LABELS[1:])
    out_df = pd.DataFrame(columns=['ImageId', 'ClassType', 'MultipolygonWKT'])

    df = pd.read_csv(os.path.join('..', 'input', 'sample_submission.csv'))
    image_ids = df['ImageId'].unique()

    for index, image_id in enumerate(image_ids):
        logger.info("-- %s" % image_id)

        labels_image_filename = get_filename(image_id, 'label')
        if not os.path.exists(labels_image_filename):
            for class_index, _ in enumerate(LABELS[1:]):
                out_df.loc[ll * index + class_index, :] = [image_id, str(class_index + 1), empty_polygon]
        else:
            labels_image = get_image_data(image_id, 'label')
            x_scaler, y_scaler = get_scalers(image_id, labels_image.shape[0], labels_image.shape[1])
            for class_index, _ in enumerate(LABELS[1:]): ## !!! for class_index in range(1, len(LABELS[1:])):
                polygons = mask_to_polygons(labels_image[:, :, class_index]) ## !!! CHECK if class_index = 0 is NONE or Buildings 
                if len(polygons) == 0:
                    out_df.loc[ll * index + class_index, :] = [image_id, str(class_index + 1), empty_polygon]
                else:
                    unit_polygons = scale(polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler, origin=(0, 0, 0))
                    out_df.loc[ll * index + class_index, :] = [image_id, str(class_index + 1), dumps(unit_polygons)]

    submission_file = '../results/submission_' + str(datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv'
    out_df.to_csv(submission_file, index=False)
    logger.info("Submission is written: %s" % submission_file)


def decision_func(img_pred):
    out = img_pred.copy()
    out[out < 0.5] = 0.0
    out[out >= 0.5] = 1.0
    return out.astype(np.uint8)


if __name__ == '__main__':

    assert len(sys.argv) == 3, "Need 2 arguments : model json file and model weights file"
    model_json_filename = sys.argv[1]
    weights_filename = sys.argv[2]
    assert os.path.exists(model_json_filename), "Model json file '%s' is not found" % model_json_filename
    assert os.path.exists(weights_filename), "Model wights file '%s' is not found" % weights_filename

    logger.info("Load trained model")
    model = load_model(model_json_filename, weights_filename)

    s = "_feature_wise"

    mean_fname = os.path.join(TRAIN_DATA, 'mean_image%s.tif' % s)
    std_fname = os.path.join(TRAIN_DATA, 'std_image%s.tif' % s)

    assert os.path.exists(mean_fname), "Mean image file '%s' is not found" % mean_fname
    assert os.path.exists(std_fname), "Std image file '%s' is not found" % std_fname

    mean_image = GeoImage(mean_fname).get_data()
    std_image = GeoImage(std_fname).get_data()

    image_ids = list(set(ALL_IMAGE_IDS) - set(TRAIN_IMAGE_IDS))
    logger.info("Compute predictions : ")
    try:
        for i, image_id in enumerate(image_ids):
            logger.info("- %s : %i / %i" % (image_id, i, len(image_ids)))
            y_predictions = compute_predictions(image_id, model, mean_image, std_image)
            y_predictions = decision_func(y_predictions)
            label_filename = get_filename(image_id, 'label')
            imwrite(label_filename, y_predictions)
    except KeyboardInterrupt:
        logger.info("Stop compute predictions")

    logger.info("Create submission")
    create_submission()






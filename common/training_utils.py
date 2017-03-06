
import numpy as np
import cv2

from data_utils import TRAIN_WKT, get_image_ids, get_filename
from geo_utils.GeoImage import GeoImage
from geo_utils.GeoImageTilers import GeoImageTilerConstSize
from image_utils import get_image_data


def normalize_image(image, mean_image, std_image):
    """
    """    
    out = image if image.dtype == np.float32 else image.astype(np.float32)
    h, w, _ = out.shape
    out -= mean_image[:h,:w,:]  # Assume that mean_image is larger or equal input image
    out /= (std_image[:h,:w,:] + 0.00001)
    return out


def tile_iterator(image_ids_to_use, 
                  channels, # input data channels to select
                  classes, # output data classes to select
                  image_type='input',
                  label_type='label',
                  balance_classes=2, # 0 = False, 1 = Method "last added", 2 = Method "equi-std", 3 = Method if only present
                  presence_percentage=2,
                  tile_size=(256, 256),
                  mean_image=None,
                  std_image=None,
                  random_rotation_angles=(0.0, 5.0, -5.0, 15.0, -15.0, 90.0, -90.0, 0.0),
                  random_scales=(),
                  resolution_levels=(1,),
                  n_images_same_time=5,
                  verbose_image_ids=False
                ):
    """
    Method returns a random tile in which at least one class of `classes` is present more than `presence_percentage`

    Random tile generation is a uniform tile selection.
    5 random images containing `classes` are selected. Overlapping tiles are searched that contain any of class.

    To uniformize tile generation, total pixel number of each class is counted and a generated tile is selected in
    a way to keep total pixel numbers balanced.


    """
    # Initialization:
    gb = TRAIN_WKT[~TRAIN_WKT['MultipolygonWKT'].str.contains("EMPTY")].groupby('ClassType')    
    image_ids = get_image_ids(classes, gb)
    image_ids = list(set(image_ids) & set(image_ids_to_use))
    step = n_images_same_time

    overlapping = int(min(tile_size[0], tile_size[1]) * 0.75)

    total_n_pixels = np.array([0] * len(classes))
    apply_random_transformation = (len(random_rotation_angles) > 0 or len(random_scales) > 0)

    if len(resolution_levels) == 0:
        resolution_levels = (1,)

    if mean_image is not None and std_image is not None:
        if len(channels) < mean_image.shape[2]:
            mean_image = mean_image[:, :, channels]
            std_image = std_image[:, :, channels]

    # Loop forever:
    while True:
        
        # Choose randomly a number (`n_images_same_time`) of images to produce tiles together
        np.random.shuffle(image_ids)
        for i, _ in enumerate(image_ids[::step]):

            e = min(step * i + step, len(image_ids))
            ids = image_ids[step*i:e]

            # Open `n_images_same_time` labels images
            gimg_tilers = []
            gimg_labels = []
            for image_id in ids:
                gimg_labels.append(GeoImage(get_filename(image_id, label_type)))
            
            # Create tile iterators: 5 x nb_levels
            for res_level in resolution_levels:
                for i in range(len(ids)):
                    gimg_label_tiles = GeoImageTilerConstSize(gimg_labels[i],
                                                              tile_size=tile_size,
                                                              scale=res_level,
                                                              min_overlapping=overlapping)
                    gimg_tilers.append(gimg_label_tiles)

            # gimg_tilers has n_images_same_time*len(resolution_levels) instances
            # gimg_tilers ~ [img1_res1, img2_res1, ..., img5_res1, img1_res2, ...]

            # Open corresponding data images
            gimg_inputs = []
            for i in ids:
                gimg_inputs.append(GeoImage(get_filename(i, image_type)))

            counter = 0
            max_counter = gimg_tilers[0].nx * gimg_tilers[0].ny
            # Iterate over all tiles of label images
            while counter < max_counter:
                all_done = True

                for tiler_index, tiles in enumerate(gimg_tilers):
                    
                    # for + break = next()
                    for tile_info_label in tiles:
                        all_done = False
                        tile_label, xoffset_label, yoffset_label = tile_info_label
                        
                        h, w, _ = tile_label.shape
                        if balance_classes > 0:
                            class_freq = np.array([0] *len(classes))
                            for ci, cindex in enumerate(classes):
                                class_freq[ci] += cv2.countNonZero(tile_label[:, :, cindex])

                            # If class representatifs are less than presence_percentage in the tile -> discard the tile
                            if np.sum(class_freq) * 100.0 / (h*w) < presence_percentage:
                               continue

                            if np.sum(total_n_pixels) > 1:
                                if balance_classes == 1:
                                    old_argmax = np.argmax(total_n_pixels)
                                    new_argmax = np.argmax(class_freq)
                                    if old_argmax == new_argmax:
                                        continue
                                elif balance_classes == 2:
                                    if np.std(total_n_pixels + class_freq) > np.std(total_n_pixels) + 200:
                                        continue
                                elif balance_classes == 3:
                                    pass
                                else:
                                    raise Exception("Method `balance_classes` is unknown")
                            total_n_pixels += class_freq
                            if verbose_image_ids:
                                print "total_n_pixels:", total_n_pixels

                        if label_type == 'label':
                            tile_label = tile_label[:, :, classes]
                            
                        #print "np.isinf(tile_label).any()", np.isinf(tile_label).any(), tile_label.min(), tile_label.max(), tile_label.shape, tile_label.dtype
                        gimg_input = gimg_inputs[tiler_index % step]
                        scale = resolution_levels[int(np.floor(tiler_index * 1.0 / step))]

                        #print "scale, xoffset_label, yoffset_label: ", scale, xoffset_label, yoffset_label
                        #print "tile_size[0], gimg_input.shape[1], gimg_input.shape[1] - xoffset_label", tile_size[0], gimg_input.shape[1], gimg_input.shape[1] - xoffset_label

                        tile_size_s = (tile_size[0]*scale, tile_size[1]*scale)
                        extent = [xoffset_label, yoffset_label, tile_size_s[0], tile_size_s[1]]
                        select_bands = None if len(channels) == gimg_input.shape[2] else channels.tolist()
                        tile_input = gimg_input.get_data(extent, *tile_size, select_bands=select_bands).astype(np.float32)
                        
                        # print "- np.isinf(tile_input).any()", np.isinf(tile_input).any(), tile_input.min(), tile_input.max(), tile_input.shape, tile_input.dtype
                        if mean_image is not None and std_image is not None:
                            # print "Extract from mean_image: ", xoffset_label, yoffset_label, tile_size_s[0], tile_size_s[1], mean_image.shape
                            mean_tile_image = mean_image[yoffset_label:yoffset_label + tile_size_s[1],
                                              xoffset_label:xoffset_label + tile_size_s[0], :]
                            std_tile_image = std_image[yoffset_label:yoffset_label + tile_size_s[1],
                                             xoffset_label:xoffset_label + tile_size_s[0], :]
                            # print "mean_tile_image.shape, std_tile_image.shape : ", mean_tile_image.shape, std_tile_image.shape
                            if scale > 1:
                                mean_tile_image = cv2.resize(mean_tile_image, dsize=tile_size, interpolation=cv2.INTER_LINEAR)
                                std_tile_image = cv2.resize(std_tile_image, dsize=tile_size, interpolation=cv2.INTER_LINEAR)
                                
                            tile_input = normalize_image(tile_input, mean_tile_image, std_tile_image)

                        #print "-- np.isinf(tile_input).any()", np.isinf(tile_input).any(), tile_input.min(), tile_input.max(), tile_input.shape, tile_input.dtype
                        # Add random rotation and scale
                        if apply_random_transformation:
                            sc = random_scales[np.random.randint(len(random_scales))] if len(random_scales) > 0 else 1.0
                            a = random_rotation_angles[np.random.randint(len(random_rotation_angles))] if len(random_rotation_angles) > 0 else 0.0
                            if 0 < np.abs(a) < 90 and sc < 1.2:
                                sc = 1.2
                            if np.abs(a) > 0.0:
                                warp_matrix = cv2.getRotationMatrix2D((tile_size[0] / 2, tile_size[1] / 2), a, sc)
                                h, w, _ = tile_input.shape
                                tile_input = cv2.warpAffine(tile_input,
                                                          warp_matrix,
                                                          dsize=(w, h),
                                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

                                tile_label = cv2.warpAffine(tile_label,
                                                          warp_matrix,
                                                          dsize=(w, h),
                                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                                if len(tile_label.shape) == 2:
                                    tile_label = np.expand_dims(tile_label, 2)
                                if len(tile_input.shape) == 2:
                                    tile_input = np.expand_dims(tile_input, 2)

                        #print "-- np.isinf(tile_label).any()", np.isinf(tile_label).any(), tile_label.min(), tile_label.max(), tile_label.shape                                            
                        #print "--- np.isinf(tile_input).any()", np.isinf(tile_input).any(), tile_input.min(), tile_input.max(), tile_input.shape
                        assert tile_label.shape[:2] == tile_input.shape[:2], "Tile sizes are not equal: {} != {}".format \
                            (tile_label.shape[:2], tile_input.shape[:2])
                        if verbose_image_ids:
                            print "Image id : ", ids[tiler_index % 5], xoffset_label, yoffset_label
                        yield tile_input, tile_label
                        break

                counter += 1
                # Check if all tilers have done the iterations
                if all_done:
                    break




def data_iterator(image_ids_to_use, 
                  channels, # input data channels to select
                  classes, # output data classes to select
                  image_type='input',
                  label_type='label',
                  balance_classes=2, # 0 = False, 1 = Method "last added", 2 = Method "equi-std", 3 = Method if only present
                  presence_percentage=2,
                  tile_size=(256, 256),
                  mean_image=None,
                  std_image=None,
                  random_rotation_angles=(0.0, 5.0, -5.0, 15.0, -15.0, 90.0, -90.0, 0.0),
                  random_scales=(),
                  resolution_levels=(1,),
                  n_images_same_time=5,
                  verbose_image_ids=False
                ):
    """
    Method returns a random tile in which at least one class of `classes` is present more than `presence_percentage`

    Random tile generation is a uniform tile selection.
    5 random images containing `classes` are selected. Overlapping tiles are searched that contain any of class.

    To uniformize tile generation, total pixel number of each class is counted and a generated tile is selected in
    a way to keep total pixel numbers balanced.


    """
    # Initialization:
    gb = TRAIN_WKT[~TRAIN_WKT['MultipolygonWKT'].str.contains("EMPTY")].groupby('ClassType')    
    image_ids = get_image_ids(classes, gb)
    image_ids = list(set(image_ids) & set(image_ids_to_use))
    step = n_images_same_time

    overlapping = int(min(tile_size[0], tile_size[1]) * 0.75)

    total_n_pixels = np.array([0] * len(classes))
    apply_random_transformation = (len(random_rotation_angles) > 0 or len(random_scales) > 0)

    if len(resolution_levels) == 0:
        resolution_levels = (1,)

    if mean_image is not None and std_image is not None:
        if len(channels) < mean_image.shape[2]:
            mean_image = mean_image[:, :, channels]
            std_image = std_image[:, :, channels]

    # Loop forever:
    while True:
        
        # Choose randomly a number (`n_images_same_time`) of images to produce tiles together
        np.random.shuffle(image_ids)
        for i, _ in enumerate(image_ids[::step]):

            e = min(step * i + step, len(image_ids))
            ids = image_ids[step*i:e]

            # Open `n_images_same_time` labels images
            gimg_tilers = []
            gimg_labels = []
            for image_id in ids:
                gimg_labels.append(GeoImage(get_filename(image_id, label_type)))
            
            # Create tile iterators: 5 x nb_levels
            for res_level in resolution_levels:
                for i in range(len(ids)):
                    gimg_label_tiles = GeoImageTilerConstSize(gimg_labels[i],
                                                              tile_size=tile_size,
                                                              scale=res_level,
                                                              min_overlapping=overlapping)
                    gimg_tilers.append(gimg_label_tiles)

            # gimg_tilers has n_images_same_time*len(resolution_levels) instances
            # gimg_tilers ~ [img1_res1, img2_res1, ..., img5_res1, img1_res2, ...]

            # Open corresponding data images
            gimg_inputs = []
            for i in ids:
                gimg_inputs.append(GeoImage(get_filename(i, image_type)))

            counter = 0
            max_counter = gimg_tilers[0].nx * gimg_tilers[0].ny
            # Iterate over all tiles of label images
            while counter < max_counter:
                all_done = True

                for tiler_index, tiles in enumerate(gimg_tilers):
                    
                    # for + break = next()
                    for tile_info_label in tiles:
                        all_done = False
                        tile_label, xoffset_label, yoffset_label = tile_info_label
                        
                        h, w, _ = tile_label.shape
                        if balance_classes > 0:
                            class_freq = np.array([0] *len(classes))
                            for ci, cindex in enumerate(classes):
                                class_freq[ci] += cv2.countNonZero(tile_label[:, :, cindex])

                            # If class representatifs are less than presence_percentage in the tile -> discard the tile
                            if np.sum(class_freq) * 100.0 / (h*w) < presence_percentage:
                               continue

                            if np.sum(total_n_pixels) > 1:
                                if balance_classes == 1:
                                    old_argmax = np.argmax(total_n_pixels)
                                    new_argmax = np.argmax(class_freq)
                                    if old_argmax == new_argmax:
                                        continue
                                elif balance_classes == 2:
                                    if np.std(total_n_pixels + class_freq) > np.std(total_n_pixels) + 200:
                                        continue
                                elif balance_classes == 3:
                                    pass
                                else:
                                    raise Exception("Method `balance_classes` is unknown")
                            total_n_pixels += class_freq
                            if verbose_image_ids:
                                print "total_n_pixels:", total_n_pixels

                        if label_type == 'label':
                            tile_label = tile_label[:, :, classes]
                            
                        #print "np.isinf(tile_label).any()", np.isinf(tile_label).any(), tile_label.min(), tile_label.max(), tile_label.shape, tile_label.dtype
                        gimg_input = gimg_inputs[tiler_index % step]
                        scale = resolution_levels[int(np.floor(tiler_index * 1.0 / step))]

                        #print "scale, xoffset_label, yoffset_label: ", scale, xoffset_label, yoffset_label
                        #print "tile_size[0], gimg_input.shape[1], gimg_input.shape[1] - xoffset_label", tile_size[0], gimg_input.shape[1], gimg_input.shape[1] - xoffset_label

                        tile_size_s = (tile_size[0]*scale, tile_size[1]*scale)
                        extent = [xoffset_label, yoffset_label, tile_size_s[0], tile_size_s[1]]
                        select_bands = None if len(channels) == gimg_input.shape[2] else channels.tolist()
                        tile_input = gimg_input.get_data(extent, *tile_size, select_bands=select_bands).astype(np.float32)
                        
                        # print "- np.isinf(tile_input).any()", np.isinf(tile_input).any(), tile_input.min(), tile_input.max(), tile_input.shape, tile_input.dtype
                        if mean_image is not None and std_image is not None:
                            # print "Extract from mean_image: ", xoffset_label, yoffset_label, tile_size_s[0], tile_size_s[1], mean_image.shape
                            mean_tile_image = mean_image[yoffset_label:yoffset_label + tile_size_s[1],
                                              xoffset_label:xoffset_label + tile_size_s[0], :]
                            std_tile_image = std_image[yoffset_label:yoffset_label + tile_size_s[1],
                                             xoffset_label:xoffset_label + tile_size_s[0], :]
                            # print "mean_tile_image.shape, std_tile_image.shape : ", mean_tile_image.shape, std_tile_image.shape
                            if scale > 1:
                                mean_tile_image = cv2.resize(mean_tile_image, dsize=tile_size, interpolation=cv2.INTER_LINEAR)
                                std_tile_image = cv2.resize(std_tile_image, dsize=tile_size, interpolation=cv2.INTER_LINEAR)
                                
                            tile_input = normalize_image(tile_input, mean_tile_image, std_tile_image)

                        #print "-- np.isinf(tile_input).any()", np.isinf(tile_input).any(), tile_input.min(), tile_input.max(), tile_input.shape, tile_input.dtype
                        # Add random rotation and scale
                        if apply_random_transformation:
                            sc = random_scales[np.random.randint(len(random_scales))] if len(random_scales) > 0 else 1.0
                            a = random_rotation_angles[np.random.randint(len(random_rotation_angles))] if len(random_rotation_angles) > 0 else 0.0
                            if 0 < np.abs(a) < 90 and sc < 1.2:
                                sc = 1.2
                            if np.abs(a) > 0.0:
                                warp_matrix = cv2.getRotationMatrix2D((tile_size[0] / 2, tile_size[1] / 2), a, sc)
                                h, w, _ = tile_input.shape
                                tile_input = cv2.warpAffine(tile_input,
                                                          warp_matrix,
                                                          dsize=(w, h),
                                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

                                tile_label = cv2.warpAffine(tile_label,
                                                          warp_matrix,
                                                          dsize=(w, h),
                                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                                if len(tile_label.shape) == 2:
                                    tile_label = np.expand_dims(tile_label, 2)
                                if len(tile_input.shape) == 2:
                                    tile_input = np.expand_dims(tile_input, 2)

                        #print "-- np.isinf(tile_label).any()", np.isinf(tile_label).any(), tile_label.min(), tile_label.max(), tile_label.shape                                            
                        #print "--- np.isinf(tile_input).any()", np.isinf(tile_input).any(), tile_input.min(), tile_input.max(), tile_input.shape
                        assert tile_label.shape[:2] == tile_input.shape[:2], "Tile sizes are not equal: {} != {}".format \
                            (tile_label.shape[:2], tile_input.shape[:2])
                        if verbose_image_ids:
                            print "Image id : ", ids[tiler_index % 5], xoffset_label, yoffset_label
                        yield tile_input, tile_label
                        break

                counter += 1
                # Check if all tilers have done the iterations
                if all_done:
                    break



def get_XY_val(val_ids,
               channels,
               labels,
               image_type='input',
               label_type='label',
               mean_image=None,
               std_image=None):
    ll = len(val_ids)
    n_channels = len(channels)
    n_labels = len(labels)

    size = [0, 0]
    for image_id in val_ids:
        image_shape = get_image_data(image_id, image_type, return_shape_only=True)
        label_image_shape = get_image_data(image_id, label_type, return_shape_only=True)
        h = max(image_shape[0], label_image_shape[0])
        w = max(image_shape[1], label_image_shape[1])
        if h > size[0]:
            size[0] = h
        if w > size[1]:
            size[1] = w

    size = tuple(size)
    X = np.zeros((ll, n_channels) + size, dtype=np.float32)
    Y = np.zeros((ll, n_labels) + size, dtype=np.float32)

    if mean_image is not None and std_image is not None:
        if n_channels < mean_image.shape[2]:
            mean_image = mean_image[:, :, channels]
            std_image = std_image[:, :, channels]

    for i, image_id in enumerate(val_ids):
        image = get_image_data(image_id, image_type).astype(np.float32)
        label_image = get_image_data(image_id, label_type).astype(np.float32)

        if n_channels < image.shape[2]:
            image = image[:, :, channels]

        if n_labels < label_image.shape[2]:
            label_image = label_image[:, :, labels]

        if mean_image is not None and std_image is not None:
            image = normalize_image(image, mean_image, std_image)

        h, w, _ = image.shape
        hl, wl, _ = label_image.shape
        image = image.transpose([2, 0, 1])
        label_image = label_image.transpose([2, 0, 1])
        X[i, :, :h, :w] = image
        Y[i, :, :hl, :wl] = label_image

    return X, Y
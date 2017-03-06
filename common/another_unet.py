#
# Another u-net from [pix2pix](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix)
#

import numpy as np
from keras.models import Model
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import Convolution2D, Convolution3D, UpSampling2D, UpSampling3D, Deconvolution2D
from keras.layers import Input, merge, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization


# def unet_zero(n_classes, n_channels, input_width, input_height, bn_mode=2, use_deconv=False):
#     nb_filters = 64
#     bn_axis = 1
#     inputs = Input(shape=(n_channels, input_height, input_width))
#
#     # 3D convolution starter:
#     x = starting_3d(inputs, nb_filters, bn_mode, bn_axis)
#
#     # Encoder:
#     _, input_height, input_width = x._keras_shape[1:] # th ordering
#     min_s = min(input_width, input_height)
#     nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
#     list_encoder, list_nb_filters = encoder(x, nb_conv, nb_filters, bn_mode, bn_axis)
#
#     #Decoder:
#     list_decoder, list_nb_filters = decoder(list_encoder, list_nb_filters, nb_filters, bn_mode, bn_axis, use_deconv)
#
#     # Termination:
#     outputs = termination(list_decoder[-1], n_classes)
#     return Model(input=inputs, output=outputs, name="U-net zero")


def unet_original(n_classes, n_channels, input_width, input_height, bn_mode=2, use_deconv=False):
    """
        Network U-Net
        Compile with Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    """
    nb_filters = 64
    bn_axis = 1
    min_size = min(input_width, input_height)
    inputs = Input(shape=(n_channels, input_height, input_width))
    
    # Encoder
    nb_conv = int(np.floor(np.log(min_size) / np.log(2)))
    list_encoder, list_nb_filters = encoder(inputs, nb_conv, nb_filters, bn_mode, bn_axis)
    # Decoder
    list_decoder, list_nb_filters = decoder(list_encoder, list_nb_filters, nb_filters, bn_mode, bn_axis, use_deconv)
    outputs = termination(list_decoder[-1], n_classes)
    return Model(input=inputs, output=outputs, name="U-net original")


def unet_3d(n_classes, n_channels, input_width, input_height, bn_mode=2):
    nb_filters = 64
    bn_axis = 1    
    inputs = Input(shape=(n_channels, input_height, input_width))

    # 2D -> 3D: 
    # Assume th ordering
    x = Reshape((1, n_channels, input_height, input_width))(inputs)

    # 3D Encoder:
    min_size = min(input_width, input_height, n_channels)
    nb_conv = int(np.floor(np.log(min_size) / np.log(2)))
    list_encoder, list_nb_filters = encoder_3d(x, nb_conv, nb_filters, bn_mode, bn_axis)
    
    # 3D Decoder:
    list_decoder, list_nb_filters = decoder_3d(list_encoder, list_nb_filters, nb_filters, bn_mode, bn_axis)

    # Termination:
    outputs = termination_3d(list_decoder[-1], n_classes)
    return Model(input=inputs, output=outputs, name="U-net 3D")


def starting_3d(input_layer, nb_filters, bn_mode, bn_axis):
    # th ordering : (None, nc, h, w)
    min_size = min(input_layer._keras_shape[1:])
    new_shape = (1,) + input_layer._keras_shape[1:]
    x_3d = Reshape(new_shape)(input_layer)    
    nb_conv = int(np.floor(np.log(min_size) / np.log(2)))
    x = x_3d
    for i in range(nb_conv):
        x = Convolution3D(nb_filters, 3, 3, 3, border_mode="same", subsample=(2,1,1))(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
        # shape = (None, nb_filters, nc / 2**i, h, w)
    x_3d = x
    # th ordering : (None, dim, 1, h, w)
    new_shape = (x_3d._keras_shape[1],) + x_3d._keras_shape[3:]
    x_2d = Reshape(new_shape)(x_3d)
    return x_2d
    

def encoder(input_layer, nb_conv, nb_filters, bn_mode, bn_axis):
    # Prepare encoder filters
    list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]
    # Encoder
    list_encoder = [Convolution2D(list_nb_filters[0], 3, 3,
                                  subsample=(2, 2), border_mode="same")(input_layer)]
    for i, f in enumerate(list_nb_filters[1:]):
        conv = conv_block_unet(list_encoder[-1], f, bn_mode, bn_axis)
        list_encoder.append(conv)

    return list_encoder, list_nb_filters


def encoder_3d(input_layer, nb_conv, nb_filters, bn_mode, bn_axis):
    # Prepare encoder filters
    list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]
    # Encoder
    list_encoder = [Convolution3D(list_nb_filters[0], 3, 3, 3,
                                  subsample=(2, 2, 2), border_mode="same")(input_layer)]
    for i, f in enumerate(list_nb_filters[1:]):
        conv = conv_3d_block_unet(list_encoder[-1], f, bn_mode, bn_axis)
        list_encoder.append(conv)

    return list_encoder, list_nb_filters


def decoder(list_encoder, list_nb_filters, nb_filters, bn_mode, bn_axis, use_deconv=False):
    # Prepare decoder filters
    nb_conv = len(list_encoder)
    list_nb_filters = list_nb_filters[:-2][::-1]
    if len(list_nb_filters) < nb_conv - 1:
        list_nb_filters.append(nb_filters)
    
    # Decoder
    _up_conv_block_unet = deconv_block_unet if use_deconv else up_conv_block_unet
    list_decoder = [_up_conv_block_unet(list_encoder[-1], list_encoder[-2],
                                       list_nb_filters[0], bn_mode, bn_axis, dropout=True)]
    for i, f in enumerate(list_nb_filters[1:]):
        # Dropout only on first few layers        
        d = True if i < 2 else False
        conv = _up_conv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, bn_mode, bn_axis, dropout=d)
        list_decoder.append(conv)
    
    return list_decoder, list_nb_filters


def decoder_3d(list_encoder, list_nb_filters, nb_filters, bn_mode, bn_axis):
    # Prepare decoder filters
    nb_conv = len(list_encoder)
    list_nb_filters = list_nb_filters[:-2][::-1]
    if len(list_nb_filters) < nb_conv-1:
        list_nb_filters.append(nb_filters)
    
    # Decoder
    list_decoder = [up_conv_3d_block_unet(list_encoder[-1], list_encoder[-2],
                                       list_nb_filters[0], bn_mode, bn_axis, dropout=True)]
    for i, f in enumerate(list_nb_filters[1:]):
        # Dropout only on first few layers        
        d = True if i < 2 else False
        conv = up_conv_3d_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, bn_mode, bn_axis, dropout=d)
        list_decoder.append(conv)
    
    return list_decoder, list_nb_filters


def termination(input_layer, n_classes):
    x = Activation("relu")(input_layer)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(n_classes, 3, 3, name="last_conv", border_mode="same")(x)
    output_layer = Activation("sigmoid")(x)
    return output_layer


def termination_3d(input_layer, n_classes):
    x = Activation("elu")(input_layer)
    x = UpSampling3D(size=(2, 2, 2))(x)
    
    # th ordering : (None, f, nc, h, w) -> (None, f*nc, h, w)
    new_shape = x._keras_shape
    n_filters = new_shape[1]*new_shape[2]
    new_shape = (n_filters, ) + new_shape[3:]
    x = Reshape(new_shape)(x)
    nb_conv = 3
    for i in range(nb_conv):
        x = Convolution2D(n_filters / 2**i, 3, 3, border_mode="same")(x)
        x = Activation("elu")(x)

    x = Convolution2D(n_classes, 3, 3, name="last_conv", border_mode="same")(x)
    output_layer = Activation("sigmoid")(x)
    return output_layer


def conv_block_unet(x, f, bn_mode, bn_axis, bn=True, subsample=(2,2)):

    x = LeakyReLU(0.2)(x)
    x = Convolution2D(f, 3, 3, subsample=subsample, border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)

    return x


def conv_3d_block_unet(x, f, bn_mode, bn_axis, bn=True, subsample=(2,2,2)):

    x = LeakyReLU(0.2)(x)
    x = Convolution3D(f, 3, 3, 3, subsample=subsample, border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)

    return x


def up_conv_block_unet(x, y, f, bn_mode, bn_axis, bn=True, dropout=False):

    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(f, 3, 3, border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = merge([x, y], mode='concat', concat_axis=bn_axis)

    return x


def up_conv_3d_block_unet(x, y, f, bn_mode, bn_axis, bn=True, dropout=False):

    x = Activation("relu")(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Convolution3D(f, 3, 3, 3, border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = merge([x, y], mode='concat', concat_axis=bn_axis)

    return x


def deconv_block_unet(x, y, f, h, w, batch_size, bn_mode, bn_axis, bn=True, dropout=False):

    o_shape = (batch_size, h * 2, w * 2, f)
    x = Activation("relu")(x)
    x = Deconvolution2D(f, 3, 3, output_shape=o_shape, subsample=(2, 2), border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = merge([x, y], mode='concat', concat_axis=bn_axis)

    return x

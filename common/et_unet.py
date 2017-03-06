
#
# ET-UNet : https://github.com/EdwardTyantov/ultrasound-nerve-segmentation/blob/master/u_model.py
#

import sys
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dense
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from metric import dice_coef, dice_coef_loss



#def unet(n_classes, n_channels, input_width, input_height):
#    inputs = Input((n_channels, IMG_ROWS, IMG_COLS), name='main_input')
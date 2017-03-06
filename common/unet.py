
from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Permute
from keras.layers import UpSampling2D, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer
from keras.constraints import nonneg


def conv(input_layer, n_filters_0=16, deep=False, l=0.01, size=3):
    """
    """
    x = Convolution2D(n_filters_0, size, size,
                      activation='relu',
                      #activation='elu',
                      init='he_normal',
                      W_regularizer=l2(l),
                      border_mode='same')(input_layer)
    if deep:
        x = Convolution2D(2 * n_filters_0, size, size,
                          activation='relu',
                          #activation='elu',
                          init='he_normal',
                          W_regularizer=l2(l),
                          border_mode='same')(x)
    return x


def unet_downsample(input_layer, n_filters_0, deep):
    """
    x -> [conv(x), conv^2(x))/2, conv^3(x))/4, conv^4(x))/8, conv^5(x)/16]
    """
    x1 = conv(input_layer, n_filters_0, deep)
    # (input_height, input_width)
    
    x = MaxPooling2D(pool_size=(2, 2))(x1)    
    x2 = conv(x, 2*n_filters_0, deep)
    # (input_height/2, input_width/2)
    
    x = MaxPooling2D(pool_size=(2, 2))(x2)
    x3 = conv(x, 4*n_filters_0, deep)
    # (input_height/4, input_width/4)    
    
    x = MaxPooling2D(pool_size=(2, 2))(x3)
    x4 = conv(x, 8*n_filters_0, deep)
    # (input_height/8, input_width/8)    
    
    x = MaxPooling2D(pool_size=(2, 2))(x4)
    x = conv(x, 16*n_filters_0, deep)
    # (input_height/16, input_width/16)
    
    return x1, x2, x3, x4, x


def unet_upsample(x1, x2, x3, x4, x5, n_filters_0, deep):
    """
    [x1, x2, x3, x4, x5] -> conv([conv([conv([conv([x5*2, x4])*2, x3])*2, x2])*2, x1])
                
    """
    x = UpSampling2D(size=(2, 2))(x5)
    x = merge([x, x4], mode='concat', concat_axis=1)
    x = conv(x, 8*n_filters_0, deep)
    # (input_height*2, input_width*2)

    x = UpSampling2D(size=(2, 2))(x)
    x = merge([x, x3], mode='concat', concat_axis=1)
    x = conv(x, 4*n_filters_0, deep)
    # (input_height*4, input_width*4)

    x = UpSampling2D(size=(2, 2))(x)
    x = merge([x, x2], mode='concat', concat_axis=1)
    x = conv(x, 2*n_filters_0, deep)
    # (input_height*8, input_width*8)

    x = UpSampling2D(size=(2, 2))(x)
    x = merge([x, x1], mode='concat', concat_axis=1)
    x = conv(x, n_filters_0, deep)
    # (input_height*16, input_width*16)
    return x
    

def unet_base(input_layer, n_filters_0, deep):    
    x1, x2, x3, x4, x = unet_downsample(input_layer, n_filters_0, deep)
    x = unet_upsample(x1, x2, x3, x4, x, n_filters_0, deep)
    return x


def original_termination(input_layer, n_classes, input_width, input_height):
    """
    U-net original termination
    """
    return Convolution2D(n_classes, 1, 1, activation='sigmoid')(input_layer)


def simple_termination(input_layer, n_classes, input_width, input_height, activation_type='sigmoid'):
    """
    
    """
    x = Convolution2D(n_classes, 1, 1)(input_layer)
    x = Reshape((n_classes, input_height * input_width))(x)
    if activation_type == 'sigmoid':
        x = Permute((2, 1))(x)
    x = Activation(activation_type)(x)
    if activation_type == 'sigmoid':
        x = Permute((2, 1))(x)
    x = Reshape((n_classes, input_height, input_width))(x)  
    return x


def unet_zero(n_classes, n_channels, input_width, input_height, deep=False, n_filters_0=32):

    inputs = Input((n_channels, input_height, input_width))
    x = unet_base(inputs, n_filters_0, deep)
    outputs = original_termination(x, n_classes, input_width, input_height)
    model = Model(input=inputs, output=outputs)
    return model


def unet_zero_prime(n_classes, n_channels, input_width, input_height, deep=False, n_filters_0=32):

    inputs = Input((n_channels, input_height, input_width))
    x = inputs
    # Mix channels
    x = conv(x, n_filters_0=n_filters_0, deep=True, l=0.0, size=1)

    x = unet_base(x, n_filters_0, deep)
    outputs = original_termination(x, n_classes, input_width, input_height)
    model = Model(input=inputs, output=outputs)
    return model


class Inverse(Layer):
    """Inverse Layer : 1/x """
    def __init__(self, **kwargs):
        super(Inverse, self).__init__(**kwargs)
        
    def call(self, x, mask=None):
        eps = K.variable(value=K.epsilon())
        x = K.pow(x + eps, -1.0)
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


class Normalization(Layer):
    """Normalization layer : x -> (x - mean)/std """
    def __init__(self, **kwargs):
        super(Normalization, self).__init__(**kwargs)
        
    def call(self, x, mask=None):
        mean = K.mean(x)
        std = K.std(x)
        x -= mean
        x /= std
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape           


def conv2(input_layer, n_filters_0=16, l=0.01):
    """
    """
    x = Convolution2D(n_filters_0, 3, 1,
                      activation='elu',
                      init='he_normal',
                      W_regularizer=l2(l),
                      border_mode='same')(input_layer)
    x = Convolution2D(2 * n_filters_0, 1, 3,
                      activation='elu',
                      init='he_normal',
                      W_regularizer=l2(l),
                      border_mode='same')(x)
    return x


def conv_downsample(input_layer, **kwargs):
    """
    """
    x = conv(input_layer, **kwargs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    return x

    
def merge_multiply(x1, x2):
    x12 = merge([x1, x2], mode='mul')
    # x12 = Normalization()(x12)
    return x12

    
def product(input_layer, base_unit, **kwargs):    
    """
    x -> (x1 * x2)
    """
    x1 = base_unit(input_layer, **kwargs)
    x2 = base_unit(input_layer, **kwargs)
    return merge_multiply(x1, x2)


def ratio(input_layer, base_unit, **kwargs):
    """
    x -> (x1 * 1/x2)
    """
    inverse_layer = Inverse()(input_layer)

    x1 = base_unit(input_layer, **kwargs)
    x2 = base_unit(inverse_layer, **kwargs)
    return merge_multiply(x1, x2)
    

def composition(input_layer, base_unit, **kwargs):
    """
    x -> x1 + (x2 * 1/x3) + (x4 * x5) + 1/x6
    """
        
    x1 = base_unit(input_layer, **kwargs)
    x23 = ratio(input_layer, base_unit, **kwargs)
    x45 = product(input_layer, base_unit, **kwargs)
    
    inverse_layer = Inverse()(input_layer)
    x6 = base_unit(inverse_layer, **kwargs)

    x16 = merge([x1, x6], mode='sum')
    x1236 = merge([x16, x23], mode='sum')
    x123456 = merge([x1236, x45], mode='sum')

    x123456 = Normalization()(x123456)
    return x123456

    
def termination(input_layer, n_classes, input_width, input_height, activation_type='sigmoid'):
    """
    
    """
    x = input_layer
    x = conv2(x, 16)
    x = conv2(x, 8)    
    return simple_termination(x, n_classes, input_width, input_height, activation_type)

    
def upsample_merge(x_small, x_large):
    x_small = UpSampling2D(size=(2, 2))(x_small)
    return merge([x_small, x_large], mode='concat', concat_axis=1)

    
def unet_one_test(n_classes, n_channels, input_width, input_height, deep=False, n_filters_0=8):

    inputs = Input((n_channels, input_height, input_width))
    x = inputs
    x = conv(x, n_filters_0=16, deep=True)    
    x = mix_channels(x, conv, n_filters_0=n_filters_0, deep=deep)    
    outputs = simple_termination(x, n_classes, input_width, input_height)    
    model = Model(input=inputs, output=outputs)
    return model


def unet_one(n_classes, n_channels, input_width, input_height, deep=False, n_filters_0=8):
    """
    Architecture:
    
    input -> [composition: conv(x) + conv(x)*conv(1/x) + conv(x)*conv(x) + conv(1/x)] -> (1), (2)
          (1) -> [composition: conv_ds(x) + conv_ds(x)*conv_ds(1/x) + conv_ds(x)*conv_ds(x) + conv_ds(1/x)] -> (3), (4)
          (3) -> [composition: conv_ds(x) + conv_ds(x)*conv_ds(1/x) + conv_ds(x)*conv_ds(x) + conv_ds(1/x)] -> (5), (6)
          (5) -> [conv] -> (7)
          (7) -> [upsample] -> (8),(6) -> [merge] -> (9)
          (9) -> [composition: conv(x) + conv(x)*conv(1/x) + conv(x)*conv(x) + conv(1/x)] -> (10)
          (10) -> [upsample] -> (11),(4) -> [merge] -> (12)
          (12) -> [composition: conv(x) + conv(x)*conv(1/x) + conv(x)*conv(x) + conv(1/x)] -> (13)
          (13) -> [conv] -> output


    !!! Ends with Nan values of loss on 10th epoch of training !!!

    """

    inputs = Input((n_channels, input_height, input_width))
    x = inputs
    
    # Downsample and store
    x = composition(x, conv, n_filters_0=n_filters_0, deep=deep)
    x0 = x
    x = composition(x, conv_downsample, n_filters_0=n_filters_0 * 2, deep=deep)
    x1 = x
    x = composition(x, conv_downsample, n_filters_0=n_filters_0 * 4, deep=deep)

    x = conv(x, n_filters_0=n_filters_0 * 8)

    # Upsample and merge
    x = upsample_merge(x, x1)
    x = composition(x, conv, n_filters_0=n_filters_0 * 4, deep=deep)
    x = upsample_merge(x, x0)
    x = composition(x, conv, n_filters_0=n_filters_0 * 2, deep=deep)

    outputs = termination(x, n_classes, input_width, input_height)
    model = Model(input=inputs, output=outputs)
    return model


def mix_channels(input_layer, base_unit, **kwargs):
    """
    x -> x1 + (x2 * 1/x3) + (x4 * x5) + 1/x6
    """        
    x1 = base_unit(input_layer, **kwargs)
    x23 = ratio(input_layer, base_unit, **kwargs)
    x45 = product(input_layer, base_unit, **kwargs)
    
    inverse_layer = Inverse()(input_layer)
    x6 = base_unit(inverse_layer, **kwargs)

    x16 = merge([x1, x6], mode='concat', concat_axis=1, name="merge_x16")
    x1236 = merge([x16, x23], mode='concat', concat_axis=1, name="merge_x1236")
    x123456 = merge([x1236, x45], mode='concat', concat_axis=1, name="merge_x123456")
    
    n_filters_0 = kwargs['n_filters_0'] if 'n_filters_0' in kwargs else 16
    x123456 = Convolution2D(n_filters_0, 1, 1,
                            activation='relu',
                            name="mix_channels_conv1D",
                            border_mode='same')(x123456)
    
    x123456 = Normalization()(x123456)
    return x123456

    
def unet_two(n_classes, n_channels, input_width, input_height, deep=False, n_filters_0=8):
    """
    Architecture:
    
    input -> [[conv]] -> [mix_channels: x + x*1/x + x*x + 1/x] -> [[U-net]] -> [termination]
   
    """

    inputs = Input((n_channels, input_height, input_width))
    x = inputs
    
    x = conv(x, n_filters_0=16, deep=False)
    
    x = mix_channels(x, conv, n_filters_0=n_filters_0, deep=deep, l=0.0)
    
    x1, x2, x3, x4, x5 = unet_downsample(x, n_filters_0, deep)
    
    x1 = conv(x1, n_filters_0=16, deep=False)
    x2 = conv(x2, n_filters_0=16, deep=False)   
    x3 = conv(x3, n_filters_0=16, deep=False)   
    x3 = conv(x3, n_filters_0=16, deep=False)          
    x4 = conv(x4, n_filters_0=16, deep=False)          
    
    x = unet_upsample(x1, x2, x3, x4, x5, n_filters_0, deep)
    
    outputs = termination(x, n_classes, input_width, input_height, activation_type='softmax')    
    model = Model(input=inputs, output=outputs)
    return model


def unet_ratios(n_classes, n_channels, input_width, input_height, deep=False, n_filters_0=8):
    """
    Architecture:

    input -> [mix_channels: x + x*1/x + x*x + 1/x] -> [[conv]] -> [[U-net]] -> [termination]

    """

    inputs = Input((n_channels, input_height, input_width))
    x = inputs

    x = ratio(x, conv, n_filters_0=n_filters_0, deep=deep, l=0.0)

    x = conv(x, n_filters_0=16, deep=False)

    x1, x2, x3, x4, x5 = unet_downsample(x, n_filters_0, deep)

    x1 = conv(x1, n_filters_0=16, deep=False)
    x2 = conv(x2, n_filters_0=16, deep=False)
    x3 = conv(x3, n_filters_0=16, deep=False)
    x3 = conv(x3, n_filters_0=16, deep=False)
    x4 = conv(x4, n_filters_0=16, deep=False)

    x = unet_upsample(x1, x2, x3, x4, x5, n_filters_0, deep)

    outputs = termination(x, n_classes, input_width, input_height, activation_type='softmax')
    model = Model(input=inputs, output=outputs)
    return model


def unet_products(n_classes, n_channels, input_width, input_height, deep=False, n_filters_0=8):
    """
    Architecture:

    input -> [mix_channels: x + x*1/x + x*x + 1/x] -> [[conv]] -> [[U-net]] -> [termination]

    """

    inputs = Input((n_channels, input_height, input_width))
    x = inputs

    x = product(x, conv, n_filters_0=n_filters_0, deep=deep, l=0.0)

    x = conv(x, n_filters_0=16, deep=False)

    x1, x2, x3, x4, x5 = unet_downsample(x, n_filters_0, deep)

    x1 = conv(x1, n_filters_0=16, deep=False)
    x2 = conv(x2, n_filters_0=16, deep=False)
    x3 = conv(x3, n_filters_0=16, deep=False)
    x3 = conv(x3, n_filters_0=16, deep=False)
    x4 = conv(x4, n_filters_0=16, deep=False)

    x = unet_upsample(x1, x2, x3, x4, x5, n_filters_0, deep)

    outputs = termination(x, n_classes, input_width, input_height, activation_type='softmax')
    model = Model(input=inputs, output=outputs)
    return model


def unet_three(n_classes, n_channels, input_width, input_height, deep=False, n_filters_0=8):
    """
    Architecture:

    input -> [mix_channels: x + x*1/x + x*x + 1/x] -> [[conv]] -> [[U-net]] -> [termination]

    """

    inputs = Input((n_channels, input_height, input_width))
    x = inputs

    x = mix_channels(x, conv, n_filters_0=n_filters_0, deep=deep, l=0.0)

    x = conv(x, n_filters_0=16, deep=False)

    x1, x2, x3, x4, x5 = unet_downsample(x, n_filters_0, deep)

    x1 = conv(x1, n_filters_0=16, deep=False)
    x2 = conv(x2, n_filters_0=16, deep=False)
    x3 = conv(x3, n_filters_0=16, deep=False)
    x3 = conv(x3, n_filters_0=16, deep=False)
    x4 = conv(x4, n_filters_0=16, deep=False)

    x = unet_upsample(x1, x2, x3, x4, x5, n_filters_0, deep)

    outputs = termination(x, n_classes, input_width, input_height, activation_type='softmax')
    model = Model(input=inputs, output=outputs)
    return model
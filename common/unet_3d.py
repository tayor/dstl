
from keras.layers import merge, Convolution3D, Convolution2D, MaxPooling3D, Input, Permute
from keras.layers import UpSampling3D, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer


def conv(input_layer, n_filters_0=16, deep=False, l=0.01, size=3):
    """
    """
    x = Convolution3D(n_filters_0, size, size, size,
                      activation='elu',
                      W_regularizer=l2(l),
                      border_mode='same')(input_layer)
    if deep:
        x = Convolution3D(2 * n_filters_0, size, size, size,
                          activation='elu',
                          W_regularizer=l2(l),
                          border_mode='same')(x)
    return x


def unet_downsample(input_layer, n_filters_0, deep, n_levels=3):
    """
    x -> [conv(x), conv^2(x))/2, conv^3(x))/4, conv^4(x))/8]
    """
    assert 0 < n_levels < 4, "Number of lever should be 1, 2, 3"
    x1 = conv(input_layer, n_filters_0, deep)
    # (n_channels, input_height, input_width)
    
    x = MaxPooling3D(pool_size=(2, 2, 2))(x1)
    x2 = conv(x, 2*n_filters_0, deep)
    # (n_channels/2, input_height/2, input_width/2)
    
    if n_levels > 1:
        x = MaxPooling3D(pool_size=(2, 2, 2))(x2)
        x3 = conv(x, 4*n_filters_0, deep)
        # (n_channels/4, input_height/4, input_width/4)
        if n_levels > 2:
            x = MaxPooling3D(pool_size=(2, 2, 2))(x3)
            x4 = conv(x, 8*n_filters_0, deep)
            # (n_channels/8 input_height/8, input_width/8)
            return x1, x2, x3, x4
        else:
            return x1, x2, x3
    else:
        return x1, x2


def unet_upsample(inputs, n_filters_0, deep):
    """
    For example
    [x1, x2, x3, x4, x5] -> conv([conv([conv([conv([x5*2, x4])*2, x3])*2, x2])*2, x1])
                
    """
    n_levels = len(inputs) - 1
    assert 0 < n_levels < 4, "Number of inputs should be 2, 3, 4"

    def _upsample_merge(_x1, _x2):
        _x = UpSampling3D(size=(2, 2, 2))(_x2)
        _x = merge([_x, _x1], mode='concat', concat_axis=1)
        return conv(_x, 4*n_filters_0, deep)
    
    x1 = inputs[0]
    x4 = None
    if n_levels > 2: 
        if x4 is None:
            x4 = inputs[3]
        x3 = inputs[2]
        x3 = _upsample_merge(x3, x4)
    else:
        x3 = None
    
    if n_levels > 1:
        if x3 is None:
            x3 = inputs[2]
        x2 = inputs[1]
        x2 = _upsample_merge(x2, x3)
    else:
        x2 = inputs[1]

    return _upsample_merge(x1, x2)
    

def unet_base(input_layer, n_filters_0, deep, n_levels=3):    
    out = unet_downsample(input_layer, n_filters_0, deep, n_levels)
    x = unet_upsample(out, n_filters_0, deep)
    return x


def original_termination(input_layer, n_classes):
    """
    U-net original termination
    """
    return Convolution2D(n_classes, 1, 1, activation='sigmoid')(input_layer)


def unet_zero(n_classes, n_channels, input_width, input_height, deep=False, n_filters_0=32, n_levels=3):

    inputs = Input((1, n_channels, input_height, input_width))
    x = unet_base(inputs, n_filters_0, deep, n_levels)
    x = conv(x, 1)
    x = Reshape((n_channels, input_height, input_width))(x)
    outputs = original_termination(x, n_classes)
    model = Model(input=inputs, output=outputs)
    return model

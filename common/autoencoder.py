
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Input, UpSampling2D, BatchNormalization, Reshape, Activation, Dropout
from keras.models import Model
from keras.regularizers import l2

def create_encoding_layers(kernel=3, filter_size=64, pad=1, pool_size=2):
    return [
        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(filter_size*2, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size*4, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size*8, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
    ]


def create_decoding_layers(kernel=3, filter_size=64, pad=1, pool_size=2):
    return[
        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(filter_size*8, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size, pool_size)),
        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(filter_size*4, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size, pool_size)),
        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(filter_size*2, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size, pool_size)),
        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
    ]


def autoencoder_zero(n_classes, n_channels, input_width, input_height, n_filters_0=64):

    inputs = Input((n_channels, input_height, input_width))

    x = inputs
    encoding_layers = create_encoding_layers(filter_size=n_filters_0)
    decoding_layers = create_decoding_layers(filter_size=n_filters_0)
    for layer in encoding_layers:
        x = layer(x)
    for layer in decoding_layers:
        x = layer(x)

    x = Convolution2D(n_classes, 1, 1, border_mode='valid',)(x)
    x = Reshape((n_classes, input_height * input_width))(x)
    x = Activation('softmax')(x)
    outputs = Reshape((n_classes, input_height, input_width))(x)

    model = Model(input=inputs, output=outputs)
    return model


def autoencoder_QBI(n_classes, n_channels, input_width, input_height, n_filters_0=16):
    """
    Inspired from https://github.com/kmader/Quantitative-Big-Imaging-2016/blob/master/Exercises/12-notebook.ipynb
    """
    
    inputs = Input((n_channels, input_height, input_width))
    x = inputs
    
    x = Convolution2D(n_filters_0, 3, 3, border_mode='same', W_regularizer=l2(l=0.02), activation='relu')(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D(pool_size=(2, 2), border_mode='same')(x)
    
    x = Convolution2D(2 * n_filters_0, 3, 3, border_mode='same', W_regularizer=l2(l=0.05), activation='relu')(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D(pool_size=(2, 2), border_mode='same')(x)

    x = Convolution2D(n_classes, 1, 1, border_mode='same', W_regularizer = l2(l=0.05))(x)

    x = UpSampling2D(size=(2,2))(x)
    x = UpSampling2D(size=(2,2))(x)
    x = Activation('sigmoid')(x)
    outputs = x
    return Model(input=inputs, output=outputs)
    
    
    




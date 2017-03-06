# https://raw.githubusercontent.com/k3nt0w/FCN_via_keras/master/FCN.py


from keras.layers import merge, Convolution2D, Deconvolution2D, MaxPooling2D, Input, Reshape, Cropping2D, Flatten
from keras.layers.core import Activation
from keras.models import Model


def fcn_zero(n_classes, nb_channels, input_width, input_height):

    FCN_CLASSES = n_classes

    # (samples, channels, rows, cols)
    inputs = Input(shape=(nb_channels, input_height, input_width))

    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #  x = input_height/2, input_width/2
    
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #  x = input_height/4, input_width/4
    
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #  x = input_height/8, input_width/8
    
    # split layer
    p3 = x
    p3 = Convolution2D(FCN_CLASSES, 1, 1, activation='relu')(p3)
    #  p3 = input_height/8, input_width/8
    #   x = input_height/8, input_width/8

    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #   x = input_height/16, input_width/16
    
    # split layer
    p4 = x
    p4 = Convolution2D(FCN_CLASSES, 1, 1, activation='relu')(p4)
    p4 = Deconvolution2D(FCN_CLASSES, 4, 4,
            output_shape=(None, FCN_CLASSES, input_height//8 + 2, input_width//8 + 2),
            subsample=(2, 2),
            border_mode='valid')(p4)
    # o = s (i - 1) + a + k - 2p, a \in {0, ..., s - 1}
    # where: i - input size (rows or cols), k - kernel size, 
    # s - stride (subsample for rows or cols respectively), p - padding size, 
    # a - user-specified quantity used to distinguish between the s different possible output sizes. Because a is not specified explicitly and Theano and Tensorflow use different values, it is better to use a dummy input and observe the actual output shape of a layer as specified in the examples.
    #
    # a = 1 for GPU Theano
    #
    # output_height = 2 * (input_height/16 - 1) + 1 + 3 - 2*0 = input_height/8 + 1 or 2
    #
    # p4 ~ input_height/8, input_width/8
    p4 = Cropping2D(cropping=((1, 1), (1, 1)))(p4)
    # p4 ~ input_height/8, input_width/8

    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #  x = input_height/32, input_width/32
    
    p5 = x
    p5 = Convolution2D(FCN_CLASSES, 1, 1, activation='relu')(p5)
    p5 = Deconvolution2D(FCN_CLASSES, 8, 8,
            output_shape=(None, FCN_CLASSES, input_height//8 + 4, input_width//8 + 4),
            subsample=(4, 4),
            border_mode='valid')(p5)
    # p5 ~ input_height/8 + 2, input_width/8 + 2
    p5 = Cropping2D(cropping=((2, 2), (2, 2)))(p5)
    # p5 = input_height/8, input_width/8
    
    # merge scores
    merged = merge([p3, p4, p5], mode='sum')
    x = Deconvolution2D(FCN_CLASSES, 16, 16,
            output_shape=(None, FCN_CLASSES, input_height + 8, input_width + 8),
            subsample=(8, 8),
            border_mode='valid')(merged)
    x = Cropping2D(cropping=((4, 4), (4, 4)))(x)
    
    output = Flatten()(x)
    output = Activation('softmax')(output)
    output = Reshape((FCN_CLASSES, input_height, input_width))(output)   
    model = Model(input=inputs, output=output)
    return model


#def to_json(model):
#    json_string = model.to_json()
#    with open('FCN_via_Keras_architecture.json', 'w') as f:
#        f.write(json_string)


# if __name__ == "__main__":
    # model = FCN()
    #visualize_model(model)
    #to_json(model)

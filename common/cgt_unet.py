#
# Code from Cogitae repository
#

from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Permute
from keras.layers import UpSampling2D, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer


def model_inputs_adapter(X):
    """
    Method to create multiple inputs for cgt_unet from 17-band X
    :param X: ndarray of shape (batch_size, n_bands, height, width)
    :returns: 
        [ 
            X[:,0,:,:], # Panchromatic
            X[:,1:3,:,:], # NDVI, GEMI
            X[:,3:5,:,:], # NDWI2, NDTI
            X[:,5:7,:,:], # BI, BI2
            X[:,7:9,:,:], # -BI, -BI2
            X[:,9:,:,:], # MS
        ]
    """
    return [ 
            X[:,0,:,:], # Panchromatic
            X[:,1:3,:,:], # NDVI, GEMI
            X[:,3:5,:,:], # NDWI2, NDTI
            X[:,5:7,:,:], # BI, BI2
            X[:,7:9,:,:], # -BI, -BI2
            X[:,9:,:,:], # MS
    ]

    
def cgt_unet_original(n_classes, n_channels, input_width, input_height, filter_size_3d=3, activation='elu'):
    
    
    # 2D convolutions before 3D convolution
    inputs = Input((n_channels, input_height, input_width))
    # Input con
    
    
    x = merge(x, mode='concat', concat_axis=1)
    pass

    
    
def __mix_input(inp1,inp2, num_filters = 3,  filter_size = 5, activation='elu'):
    merged = merge([inp1,inp2], mode='concat', concat_axis=1)
    inp = Reshape( (1,) + merged._keras_shape[1:])(merged)
    num_layers = inp._keras_shape[2]
    conv = Convolution3D(num_filters, num_layers, filter_size, filter_size, activation=activation,border_mode='same')(inp)
    pool = MaxPooling3D(pool_size=(num_layers, 1, 1))(conv)
    out = Reshape( (num_filters,) + merged._keras_shape[2:])(pool)
    return out
    
    
def __preproc_3D(band, filter_size, lisz,  dim_ordering, activation='elu' ):
    # 
    # num_bands_by_img = {
    #   '3' : 3,
    #   'P' : 1,
    #   'M' : 8,
    #   'A' : 8,
    # }
    
    num_in_band = num_bands_by_img[band]
    if dim_ordering == 'th':
        inp_shape = ( num_in_band, lisz, lisz)
    else:
        inp_shape = ( lisz, lisz, num_in_band)
    linput = Input(inp_shape,name='input_{}'.format(band))
    convpre = Convolution2D(num_in_band, filter_size, filter_size, activation=activation, border_mode='same' )(linput)
#        poolpre = MaxPooling2D(pool_size=(2, 2))(convpre)
#        poolpre = Dropout(0.5)(poolpre)
    convpre = Convolution2D(num_in_band, filter_size, filter_size, activation=activation, border_mode='same' )(convpre)
    return linput, convpre

    
def __get_unet_3D_Mix( ISZ = ISZ, N_Cls=N_Cls, bands_used = bands_used, activation = 'elu', filter_size_3D = 3):
    
    # ISZ = 80
    # N_Cls = 10
    # bands_used = [ '3', 'P', 'M', 'A' ]

    dim_ordering = K.image_dim_ordering()
    
    lisz = ISZ
    
    input_dict = {}
    inputs = []
    first_convs = []

    for band in bands_used:
        linput, convpre = preproc_3D(band, filter_size_3D, lisz=lisz, dim_ordering=dim_ordering, activation='elu')
        input_dict[band] = linput
        inputs.append(linput)
        first_convs.append(convpre)

    convAB = mix_input(input_dict['A'],input_dict['M'], num_filters = 3,  filter_size = 5)
    first_convs.append(convAB)
    N_Bands = 20 + 3

    uppre = merge(first_convs, mode='concat', concat_axis=1)
    
#    print('uppre shape : {}'.format(uppre._keras_shape))
    inputs3D = Reshape((1,N_Bands, lisz, lisz))(uppre)
    conv1 = Convolution3D(1, N_Bands, 3, 3, activation=activation, border_mode='same' )(inputs3D)
    pool0 = Reshape((N_Bands, lisz, lisz))(conv1)    
    
    conv1, pool1 = conv_down(pool0, 32 )
    # size / 2
    conv2, pool2 = conv_down(pool1, 64 )
    # size / 4
    conv3, pool3 = conv_down(pool2, 128 )
    # size / 8
    conv4, pool4 = conv_down(pool3, 256 )
    # size / 16

    # bottom some convolutions to mix features

    conv5 = Convolution2D(512, 3, 3, activation=activation, border_mode='same' )(pool4)
    conv5 = Convolution2D(512, 3, 3, activation=activation, border_mode='same' )(conv5)

    # size / 16
    conv6 = conv_up(conv5, conv4, 256 )
    # size / 8
    conv7 = conv_up(conv6, conv3, 128 )
    # size / 4
    conv8 = conv_up(conv7, conv2, 64 )
    # size / 2
    conv9 = conv_up(conv8, conv1, 32 )
    # size / 1

    conv10 = Convolution2D(N_Cls, 1, 1, activation='sigmoid' )(conv9)

#    outputs = [conv10, test]
    outputs = [conv10]
    
    model = Model(input=inputs, output=outputs)

    model_name = 'unet_3D_mix_{0}{0}'.format(filter_size_3D)
    with open(os.path.join(modelDir, 'model_{}_{}_{}_{}_{}.json'.format(model_name, ISZ, N_Cls, dim_ordering, '_'.join(bands_used))), 'w') as modelf:
        json_string = model.to_json()
        modelf.write(json_string)
    return model

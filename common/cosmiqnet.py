#
# Cosmiq network inspired from https://gist.github.com/hagerty/5e1fb0eef76553f7d26dfb4d136b3443
#

from keras.models import Model
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers import Input, merge, Lambda
from keras.layers.advanced_activations import LeakyReLU


def cosmiqnet_original(input_a_shape, input_b_shape, n_layers=5, n_filters=64, size=3, beta=0.9):
    """
    
    :param input_a_shape: shape of larger size input, (h1, w1, nc1)
    :param input_b_shape: shape of smaller size input, (h2, w2, nc2) such that h2 = h1 / scale, w2 = w1 / scale
    :param output_shape:
    """
    assert input_a_shape[0] * 1.0 / input_b_shape[0] == input_a_shape[1] * 1.0 / input_b_shape[1], "Inputs image size should be proportional"
    x_a = Input(shape=(input_a_shape[0],) + input_a_shape[:2])
    x_b = Input(shape=(input_b_shape[0],) + input_b_shape[:2])
    
    scale = int(input_a_shape[0] * 1.0 / input_b_shape[0])

    # first layer
    z_1 = mix_block(x_a, x_b, n_filters, size, scale)
    label_out = internal_block(z_1, n_filters, size)

    # Loop
    x = label_out
    for i in range(n_layers - 1):
        z_i = mix_block(x_a, x_b, n_filters, size, scale)
        x = Convolution2D(n_filters, size, size, border_mode="same")(x)
        x = merge([x, z_i], mode='sum')
        x = internal_block(x, n_filters, size)
        label_out = beta_mix(x, label_out, beta)
        x = label_out

    return Model(input=[x_a, x_b], output=label_out)


def cosmiqnet_zero(input_shape, output_n_channels, n_layers=5, n_filters=64, size=3, beta=0.9):
    """
    """
    inputs = Input(shape=(input_shape[2],) + input_shape[:2])

    # first layer
    z_1 = Convolution2D(n_filters, size, size, border_mode="same")(inputs)
    label_out = internal_block(z_1, n_filters, size)

    # Loop
    out_layers = [label_out,]
    for i in range(1, n_layers):
        z_i = Convolution2D(n_filters, size, size, border_mode="same")(inputs)
        x = Convolution2D(n_filters, size, size, border_mode="same")(out_layers[i-1])
        x = merge([x, z_i], mode='sum')
        x = internal_block(x, n_filters, size)
        label_out = beta_mix(x, out_layers[i-1], beta)
        out_layers.append(label_out)

    outputs = Convolution2D(output_n_channels, 1, 1, border_mode="same")(out_layers[-1])

    return Model(input=inputs, output=outputs)


def mix_block(x_a, x_b, n_filters, size, scale):
    z_a = Convolution2D(n_filters, size, size, subsample=(scale, scale), border_mode="same")(x_a)
    z_b = Convolution2D(n_filters, size, size, border_mode="same")(x_b)
    return merge([z_a, z_b], mode='sum')


def internal_block(x, n_filters, size):
    x = LeakyReLU()(x)
    x = Convolution2D(n_filters, size, size, border_mode="same")(x)
    x = LeakyReLU()(x)
    x = Deconvolution2D(1, size, size, output_shape=x._keras_shape, border_mode="same")(x)
    return x


def beta_mix(x, y, beta):
    # x = Lambda(lambda _x: beta * _x, output_shape=lambda input_shape: input_shape)(x)
    # y = Lambda(lambda _y: (1.0 - beta) * _y, output_shape=lambda input_shape: input_shape)(y)
    return merge([x, y], mode='sum')


def conv_mix(x, y):
    # x = Lambda(lambda _x: beta * _x, output_shape=lambda input_shape: input_shape)(x)
    # y = Lambda(lambda _y: (1.0 - beta) * _y, output_shape=lambda input_shape: input_shape)(y)
    xy = merge([x, y], mode='concat', concat_axis=1)
    return Convolution2D(x._keras_shape[1], 1, 1, border_mode="same")(xy)



        # with tf.device(gpu):
#     # Generator
#     x8 = tf.placeholder(tf.float32, shape=[None, FLAGS.ws, FLAGS.ws, 8])
#     x3 = tf.placeholder(tf.float32, shape=[None, scale * FLAGS.ws, scale * FLAGS.ws, 3])
#     label_distance = tf.placeholder(tf.float32, shape=[None, FLAGS.ws, FLAGS.ws, 1])
#     for i in range(layers):
#         alpha[i] = tf.Variable(0.9, name='alpha_' + str(i))
#         beta[i] = tf.maximum(0.0, tf.minimum(1.0, alpha[i]), name='beta_' + str(i))
#         bi[i] = tf.Variable(tf.constant(0.0, shape=[FLAGS.filters]), name='bi_' + str(i))
#         bo[i] = tf.Variable(tf.constant(0.0, shape=[FLAGS.filters]), name='bo_' + str(i))
#         Wo[i] = tf.Variable(
#             tf.truncated_normal([FLAGS.filter_size, FLAGS.filter_size, 1, FLAGS.filters], stddev=0.1),
#             name='Wo_' + str(i))  #
#         Wi3[i] = tf.Variable(
#             tf.truncated_normal([FLAGS.filter_size, FLAGS.filter_size, 3, FLAGS.filters], stddev=0.1),
#             name='Wi_' + str(i) + 'l3')
#         Wi8[i] = tf.Variable(
#             tf.truncated_normal([FLAGS.filter_size, FLAGS.filter_size, 8, FLAGS.filters], stddev=0.1),
#             name='Wi_' + str(i) + 'l8')
#         z3[i] = tf.nn.conv2d(x3, Wi3[i], strides=[1, scale, scale, 1], padding='SAME')
#         z8[i] = tf.nn.conv2d(x8, Wi8[i], strides=[1, 1, 1, 1], padding='SAME')
#         if 0 == i:
#             z[i] = tf.nn.bias_add(tf.nn.relu(tf.nn.bias_add(tf.add(z3[i], z8[i]), bi[i], name='conv_' + str(i))),
#                                   bo[i])
#         else:
#             inlayer[i] = outlayer[i - 1]
#             Wi[i] = tf.Variable(
#                 tf.truncated_normal([FLAGS.filter_size, FLAGS.filter_size, 1, FLAGS.filters], stddev=0.1),
#                 name='Wi_' + str(i))
#             z[i] = tf.nn.bias_add(tf.nn.relu(tf.nn.bias_add(
#                 tf.add(tf.add(z3[i], z8[i]), tf.nn.conv2d(inlayer[i], Wi[i], strides=[1, 1, 1, 1], padding='SAME')),
#                 bi[i], name='conv_' + str(i))), bo[i])
#         Wii[i] = tf.Variable(
#             tf.truncated_normal([FLAGS.filter_size, FLAGS.filter_size, FLAGS.filters, FLAGS.filters], stddev=0.1),
#             name='Wii_' + str(i))
#         bii[i] = tf.Variable(tf.constant(0.0, shape=[FLAGS.filters]), name='bii_' + str(i))
#         zz[i] = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(z[i], Wii[i], strides=[1, 1, 1, 1], padding='SAME'), bii[i]))
#         labelout[i] = tf.nn.conv2d_transpose(zz[i], Wo[i], [FLAGS.batch_size, FLAGS.ws, FLAGS.ws, 1],
#                                              strides=[1, 1, 1, 1], padding='SAME')
#         if 0 == i:
#             outlayer[i] = labelout[i]
#         else:
#             outlayer[i] = tf.nn.relu(
#                 tf.add(tf.scalar_mul(beta[i], labelout[i]), tf.scalar_mul(1.0 - beta[i], inlayer[i])))
#         label_cost[i] = tf.reduce_sum(tf.pow(tf.sub(outlayer[i], label_distance), 2))
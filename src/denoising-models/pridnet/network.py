import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D

def lrelu(x):
    return tf.maximum(x * 0.2, x)

def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.random.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output

def unet(input):
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same')(input)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same')(conv1)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same')(conv1)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(conv1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, padding='same')(conv2)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, padding='same')(conv2)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(conv2)

    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, padding='same')(conv3)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, padding='same')(conv3)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(conv3)

    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, padding='same')(conv4)
    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, padding='same')(conv4)
    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(conv4)

    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation=lrelu, padding='same')(pool4)
    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation=lrelu, padding='same')(conv5)
    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation=lrelu, padding='same')(conv5)
    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation=lrelu, padding='same')(conv5)

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, padding='same')(up6)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, padding='same')(conv6)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, padding='same')(conv6)

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, padding='same')(up7)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, padding='same')(conv7)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, padding='same')(conv7)

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, padding='same')(up8)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, padding='same')(conv8)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, padding='same')(conv8)

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same')(up9)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same')(conv9)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same')(conv9)

    conv10 = tf.keras.layers.Conv2D(1, (1, 1), activation=None)(conv9)
    
    return conv10

def feature_encoding(input):
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same', name='fe_conv1')(input)
    conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same', name='fe_conv2')(conv1)
    conv3 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same', name='fe_conv3')(conv2)
    conv4 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same', name='fe_conv4')(conv3)
    conv4 = squeeze_excitation_layer(conv4, 32, 2)
    output = tf.keras.layers.Conv2D(1, (3, 3), activation=lrelu, padding='same', name='fe_conv5')(conv4)

    return output

def avg_pool(feature_map):
    ksize = [[1, 1, 1, 1], [1, 2, 2, 1], [1, 4, 4, 1], [1, 8, 8, 1], [1, 16, 16, 1]]
    pool1 = tf.nn.avg_pool(feature_map, ksize=ksize[0], strides=ksize[0], padding='VALID')
    pool2 = tf.nn.avg_pool(feature_map, ksize=ksize[1], strides=ksize[1], padding='VALID')
    pool3 = tf.nn.avg_pool(feature_map, ksize=ksize[2], strides=ksize[2], padding='VALID')
    pool4 = tf.nn.avg_pool(feature_map, ksize=ksize[3], strides=ksize[3], padding='VALID')
    pool5 = tf.nn.avg_pool(feature_map, ksize=ksize[4], strides=ksize[4], padding='VALID')

    return pool1, pool2, pool3, pool4, pool5

def all_unet(pool1, pool2, pool3, pool4, pool5):
    unet1 = unet(pool1)
    unet2 = unet(pool2)
    unet3 = unet(pool3)
    unet4 = unet(pool4)
    unet5 = unet(pool5)

    return unet1, unet2, unet3, unet4, unet5

def resize_all_image(unet1, unet2, unet3, unet4, unet5):
    resize1 = tf.image.resize(unet1, [tf.shape(unet1)[1], tf.shape(unet1)[2]], method=tf.image.ResizeMethod.BILINEAR)
    resize2 = tf.image.resize(unet2, [tf.shape(unet1)[1], tf.shape(unet1)[2]], method=tf.image.ResizeMethod.BILINEAR)
    resize3 = tf.image.resize(unet3, [tf.shape(unet1)[1], tf.shape(unet1)[2]], method=tf.image.ResizeMethod.BILINEAR)
    resize4 = tf.image.resize(unet4, [tf.shape(unet1)[1], tf.shape(unet1)[2]], method=tf.image.ResizeMethod.BILINEAR)
    resize5 = tf.image.resize(unet5, [tf.shape(unet1)[1], tf.shape(unet1)[2]], method=tf.image.ResizeMethod.BILINEAR)

    return resize1, resize2, resize3, resize4, resize5

def to_clean_image(feature_map, resize1, resize2, resize3, resize4, resize5):
    concat = tf.concat([feature_map, resize1, resize2, resize3, resize4, resize5], 3)
    sk_conv1 = tf.keras.layers.Conv2D(7, (3, 3), activation=lrelu, padding='same')(concat)
    sk_conv2 = tf.keras.layers.Conv2D(7, (5, 5), activation=lrelu, padding='same')(concat)
    sk_conv3 = tf.keras.layers.Conv2D(7, (7, 7), activation=lrelu, padding='same')(concat)
    sk_out = selective_kernel_layer(sk_conv1, sk_conv2, sk_conv3, 4, 7)
    output = tf.keras.layers.Conv2D(1, (3, 3), activation=None, padding='same')(sk_out)

    return output

def squeeze_excitation_layer(input_x, out_dim, middle):
    squeeze = GlobalAveragePooling2D()(input_x)
    excitation = tf.keras.layers.Dense(middle, use_bias=True, activation='relu')(squeeze)
    excitation = tf.keras.layers.Dense(out_dim, use_bias=True, activation='sigmoid')(excitation)
    excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
    scale = input_x * excitation
    return scale

def selective_kernel_layer(sk_conv1, sk_conv2, sk_conv3, middle, out_dim):
    sum_u = sk_conv1 + sk_conv2 + sk_conv3
    squeeze = GlobalAveragePooling2D()(sum_u)
    squeeze = tf.reshape(squeeze, [-1, 1, 1, out_dim])
    z = tf.keras.layers.Dense(middle, use_bias=True, activation='relu')(squeeze)
    a1 = tf.keras.layers.Dense(out_dim, use_bias=True)(z)
    a2 = tf.keras.layers.Dense(out_dim, use_bias=True)(z)
    a3 = tf.keras.layers.Dense(out_dim, use_bias=True)(z)

    before_softmax = tf.concat([a1, a2, a3], 1)
    after_softmax = tf.nn.softmax(before_softmax, axis=1)
    a1 = after_softmax[:, 0, :, :]
    a1 = tf.reshape(a1, [-1, 1, 1, out_dim])
    a2 = after_softmax[:, 1, :, :]
    a2 = tf.reshape(a2, [-1, 1, 1, out_dim])
    a3 = after_softmax[:, 2, :, :]
    a3 = tf.reshape(a3, [-1, 1, 1, out_dim])

    select_1 = sk_conv1 * a1
    select_2 = sk_conv2 * a2
    select_3 = sk_conv3 * a3

    out = select_1 + select_2 + select_3

    return out

def network(in_image):
    feature_map = feature_encoding(in_image)
    feature_map_2 = tf.concat([in_image, feature_map], 3)
    pool1, pool2, pool3, pool4, pool5 = avg_pool(feature_map_2)
    unet1, unet2, unet3, unet4, unet5 = all_unet(pool1, pool2, pool3, pool4, pool5)
    resize1, resize2, resize3, resize4, resize5 = resize_all_image(unet1, unet2, unet3, unet4, unet5)
    out_image = to_clean_image(feature_map_2, resize1, resize2, resize3, resize4, resize5)

    return out_image

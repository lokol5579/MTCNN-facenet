from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Input
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Model
import keras.backend as K

def relu6(x):
    return K.relu(x=x, max_value=6)

def conv(inputs, filters, kernel_size=3, strides=1):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)
    return x

def conv_ds(inputs, pointwise_conv_filters, depth_multiplier=1, strides=1):
    x = DepthwiseConv2D(filters=3, depth_multiplier=depth_multiplier, strides=strides, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)

    x = conv(inputs=x, filters=pointwise_conv_filters, kernel_size=1, strides=1)
    return x

def MobileNet(input_shape=(160, 160, 3), embedding=128, dropout_prob=0.4, depth_multiplier=1):
    inputs = Input(shape=input_shape)

    # 160,160,3 -> 80,80,32
    x = conv(inputs, filters=32, strides=2)

    # 80,80,32 -> 80,80,64
    x = conv_ds(x, pointwise_conv_filters=64, depth_multiplier=depth_multiplier)

    # 80,80,64 -> 40,40,128
    x = conv_ds(x, pointwise_conv_filters=128, depth_multiplier=depth_multiplier, strides=2)
    x = conv_ds(x, pointwise_conv_filters=128, depth_multiplier=depth_multiplier)
    x = conv_ds(x, pointwise_conv_filters=128, depth_multiplier=depth_multiplier)
    x = conv_ds(x, pointwise_conv_filters=128, depth_multiplier=depth_multiplier)
    x = conv_ds(x, pointwise_conv_filters=128, depth_multiplier=depth_multiplier)
    x = conv_ds(x, pointwise_conv_filters=128, depth_multiplier=depth_multiplier)

    # 40,40,128 -> 20,20,256
    x = conv_ds(x, pointwise_conv_filters=256, depth_multiplier=depth_multiplier, strides=2)
    x = conv_ds(x, pointwise_conv_filters=256, depth_multiplier=depth_multiplier)
    x = conv_ds(x, pointwise_conv_filters=256, depth_multiplier=depth_multiplier)
    x = conv_ds(x, pointwise_conv_filters=256, depth_multiplier=depth_multiplier)
    x = conv_ds(x, pointwise_conv_filters=256, depth_multiplier=depth_multiplier)
    x = conv_ds(x, pointwise_conv_filters=256, depth_multiplier=depth_multiplier)

    # 20,20,256 -> 10,10,512
    x = conv_ds(x, pointwise_conv_filters=512, depth_multiplier=depth_multiplier, strides=2)
    x = conv_ds(x, pointwise_conv_filters=512, depth_multiplier=depth_multiplier)
    x = conv_ds(x, pointwise_conv_filters=512, depth_multiplier=depth_multiplier)
    x = conv_ds(x, pointwise_conv_filters=512, depth_multiplier=depth_multiplier)
    x = conv_ds(x, pointwise_conv_filters=512, depth_multiplier=depth_multiplier)
    x = conv_ds(x, pointwise_conv_filters=512, depth_multiplier=depth_multiplier)

    # 10,10,512 -> 5,5,1024
    x = conv_ds(x, pointwise_conv_filters=1024, depth_multiplier=depth_multiplier, strides=2)
    x = conv_ds(x, pointwise_conv_filters=1024, depth_multiplier=depth_multiplier)

    # 1024 Pooling
    x = AveragePooling2D()(x)

    # dropout
    x = Dropout(1 - dropout_prob)(x)

    # 全连接
    x = Dense(embedding, use_bias=False)(x)
    x = BatchNormalization(momentum=0.99, epsilon=10e-6, scale=False)(x)

    # 创建模型
    model = Model(inputs, x)

    return model

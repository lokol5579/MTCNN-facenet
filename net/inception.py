from keras import backend
from keras import Model
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import add

def conv_bn(x, filters, kernel_size, strides=1, padding='valid', activation='relu', use_bias=False):
    #创建卷积层
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
    #由于进行归一化，有无偏置都没有影响
    if not use_bias:
        x = BatchNormalization(axis=3, momentum=0.99, epsilon=10e-6, scale=False)(x)
    if activation is not None:
        x = Activation(activation=activation)(x)
    return x

def scaling(x, scale):
    return x * scale

def InceptionresnetA(x, scale):
    x = Activation(activation='relu')(x)

    branch1 = conv_bn(x=x, filters=32, kernel_size=1, padding='same')

    branch2 = conv_bn(x=x, filters=32, kernel_size=1, padding='same')
    branch2 = conv_bn(x=branch2, filters=32, kernel_size=3, padding='same')

    branch3 = conv_bn(x=x, filters=32, kernel_size=1, padding='same')
    branch3 = conv_bn(x=branch3, filters=32, kernel_size=2, padding='same')
    branch3 = conv_bn(x=branch3, filters=32, kernel_size=3, padding='same')

    branch = Concatenate(axis=3)([branch1, branch2, branch3])

    branch0 = conv_bn(x=branch, filters=backend.int_shape(x)[3], kernel_size=1, activation=None, use_bias=True)
    #此处有问题
    branch0 = Lambda(scaling, output_shape=backend.int_shape(branch0)[1:], arguments={'scale': scale})(branch0)

    x = add([x, branch0])

    x = Activation(activation='relu')(x)

    return x

def InceptionResNetV1(input_shape=(160, 160, 3), classes=128, dropout_keep_prob=0.8):
    channel_num = 3

    # stem
    input = Input(shape=input_shape)

    #160, 160, 3 -> 77, 77, 64
    x = conv_bn(x=input, filters=32, kernel_size=3, strides=2)
    x = conv_bn(x=x, filters=32, kernel_size=3)
    x = conv_bn(x=x, filters=64, kernel_size=3, padding='same')

    # 77, 77, 64 -> 38, 38, 64
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # 38,38,64 -> 17,17,256
    x = conv_bn(x=x, filters=80, kernel_size=1)
    x = conv_bn(x=x, filters=192, kernel_size=3)
    x = conv_bn(x=x, filters=256, kernel_size=3, strides=2)

    #5 * Inception-resnet-A
    for i in range(0, 5):
        InceptionresnetA(x=x, scale=0.17)

    # Reduction-A
    # 17,17,256 -> 8, 8, 896
    branch0 = conv_bn(x, )





    # model = Model(input, x)
    # model.summary()


# if __name__ == "__main__":
#     InceptionResNetV1()



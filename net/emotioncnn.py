from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import SeparableConv2D
from keras.layers import Input
from keras.layers import Add
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
import keras.backend as K


def relu6(x):
    return K.relu(x=x, max_value=6)

def conv(inputs, filters, kernel_size=3, strides=1, padding='valid',layer_num=0, activate=True):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False, name="conv_%d"%layer_num)(inputs)
    x = BatchNormalization(name="conv_batchnormalization_%d"%layer_num)(x)
    if activate:
        x = Activation(relu6, name="conv_activation_%d"%layer_num)(x)
    return x, layer_num + 1

def conv_separable(inputs, filters, kernel_size=3, strides=1, layer_num=0, activate=True):
    x = SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, name="sc_conv_%d"%layer_num)(inputs)
    x = BatchNormalization(name="sc_conv_batchnormalization_%d"%layer_num)(x)
    if activate:
        x = Activation(relu6, name="sc_conv_activation_%d"%layer_num)(x)
    return x, layer_num + 1

def EmotionCNN(inputs, embedding=128, dropout_keep_prob=0.4, depth_multiplier=1, layer_num=1):
    #64,64,1 -> 62,62,8
    x, layer_num = conv(inputs, filters=8, layer_num=layer_num)
    #62,62,8 -> 60,60,8
    x, layer_num = conv(x, filters=8, layer_num=layer_num)

    #60,60,8 -> 30,30,16
    branch1, layer_num = conv_separable(x, filters=16, layer_num=layer_num)
    branch1, layer_num = conv_separable(branch1, filters=16, layer_num=layer_num, activate=False)
    branch1 = MaxPooling2D(name="max_pooling_%d"%(layer_num-1))(branch1)

    branch2, layer_num = conv(x, filters=16, kernel_size=1, strides=2, padding='same', layer_num=layer_num, activate=False)

    x = Add(name="add_1")([branch1, branch2])

    #30,30,16 -> 15,15,32
    branch1, layer_num = conv_separable(x, filters=32, layer_num=layer_num)
    branch1, layer_num = conv_separable(branch1, filters=32, layer_num=layer_num, activate=False)
    branch1 = MaxPooling2D(name="max_pooling_%d"%(layer_num-1))(branch1)
    branch2, layer_num = conv(x, filters=32, kernel_size=1, strides=2, padding='same', layer_num=layer_num, activate=False)
    x = Add(name="add_2")([branch1, branch2])

    #15,15,32 -> 8,8,64
    branch1, layer_num = conv_separable(x, filters=64, layer_num=layer_num)
    branch1, layer_num = conv_separable(branch1, filters=64, layer_num=layer_num, activate=False)
    branch1 = MaxPooling2D(padding="same", name="max_pooling_%d"%(layer_num-1))(branch1)
    branch2, layer_num = conv(x, filters=64, kernel_size=1, strides=2, padding='same', layer_num=layer_num, activate=False)
    x = Add(name="add_3")([branch1, branch2])

    #8,8,64 -> 4,4,128
    branch1, layer_num = conv_separable(x, filters=128, layer_num=layer_num)
    branch1, layer_num = conv_separable(branch1, filters=128, layer_num=layer_num, activate=False)
    branch1 = MaxPooling2D(name="max_pooling_%d"%(layer_num-1))(branch1)
    branch2, layer_num = conv(x, filters=128, kernel_size=1, strides=2, padding='same', layer_num=layer_num, activate=False)
    x = Add(name="add_4")([branch1, branch2])

    x = Conv2D(filters=7, kernel_size=3, padding='same', use_bias=True, name="conv_%d"%layer_num)(x)
    x = GlobalAveragePooling2D(name="global_average_pooling_%d"%layer_num)(x)
    x = Activation(relu6, name="predictions")(x)

    model = Model(inputs, x)

    #model.summary()
    return model


# if __name__ == "__main__":
#     # model = load_model("./models/emotion_model.hdf5")
#     # model.summary()
#     # model.save_wights("./models/emotion_model_wights.h5")
#
#     inputs = Input([64, 64, 1], name="Input_0")
#     model = EmotionCNN(inputs)
#     # model.save_weights("./models/emotion_model_wights.h5")
#     print(type(model))
#     model.load_weights("./models/emotion_model.hdf5")
#     model.save_weights("./models/emotion_model_wights.h5")
#     print(type(model))
#     # model.save_wights("./models/emotion_model_wights.h5")



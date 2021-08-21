import keras
import keras.backend as K
from keras.layers import Lambda
from keras.layers import Input
from keras.models import Model
from net.mobilenet import MobileNet

def facenet(input_shape):
    inputs = Input(shape=input_shape, name="mobilenet_input")
    model = MobileNet(inputs)
    x = Lambda(lambda  x: K.l2_normalize(x, axis=1), name="Embedding")(model.output)
    model = Model(inputs, x)
    # model.summary()
    return model

#if __name__ == "__main__":
#    input_shape = [160,160,3]
#    facenet(input_shape)
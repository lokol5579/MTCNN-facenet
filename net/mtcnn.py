from keras.layers import Conv2D, Input, MaxPool2D, Reshape, Activation, Flatten, Dense, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model, Sequential
import tensorflow as tf
import numpy as np
import utils.utils as utils
import cv2


# -----------------------------#
#   粗略获取人脸框
#   输出bbox位置和是否有人脸
# -----------------------------#
def create_Pnet(weight_path):
    # h,w
    # 初始化深度学习网络输入层的tensor,本句最终返回的tensor为(None,None,None,3)第一个None是一次训练所抓取的数据样本量
    # 由于会输入不同大小的图片,所以input长和宽没有指定
    input = Input(shape=[None, None, 3])  # shape表示张量的维度的元组

    # h,w,3 -> h/2,w/2,10
    # 经过一次卷积
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    # 在x<0的情况下ReLU依然有输出
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    # 整张图片变为原来长和宽的1/2
    # 池化
    x = MaxPool2D(pool_size=2)(x)

    # h/2,w/2,10 -> h/2,w/2,16
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)
    # h/2,w/2,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    # h/2, w/2, 2(特征层)
    # 原图长为h,宽为w,把整张图片划分成了2/h*2/w的网格,classifier代表这块区域有人脸的置信度
    # 进行一个双通道的卷积
    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    # 无激活函数，线性。
    # h/2, w/2, 4(特征层)
    # 进行一个4通道的卷积
    # bbox_regress代表右下角区域这个人脸框的位置
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model
    # 经过PNet后得到一大堆人脸框和置信度
    # PNet得到的所有候选框都会在原图中截下小图片后传入到RNet中
    # RNet再对图片进行处理,判断这张图片里面是否存在人脸,并且对框进行调整,使其更符合人脸这样一个形状


# -----------------------------#
#   mtcnn的第二段
#   精修框
# -----------------------------#
def create_Rnet(weight_path):
    # 把图片重新调整大小变为24*24*3
    input = Input(shape=[24, 24, 3])
    # 24,24,3 -> 11,11,28
    # 28通道的卷积
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # 11,11,28 -> 4,4,48
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 4,4,48 -> 3,3,64
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    # 3,3,64 -> 64,3,3
    # 进行通道的切换,把第一个维度和第三个维度进行切换,输出变成64*3*3
    x = Permute((3, 2, 1))(x)
    # 铺平,得到一个576维的向量
    x = Flatten()(x)
    # 576 -> 128
    # 576维的向量经过一个全连接层得到128维的向量
    x = Dense(128, name='conv4')(x)
    x = PReLU(name='prelu4')(x)
    # 128 -> 2 128 -> 4
    # 经过两个全连接层映射到classifier和bbox_regress上
    # RNet的传入是已经确定好的一张图片,RNet只要判断这张图片有么有人脸,在有人脸时利用bbox_regress得到的四个参数对图片的长宽位置进行调整
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = Dense(4, name='conv5-2')(x)
    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model
    # RNet输出是每一个框的修正值,把一些没有人脸的框进行剔除
    # 在RNet之后得到一些框,把它截下来再传入到ONet中,ONet对图片进行识别
    # ONet也是判断截下来的图中是否有人脸,如果有人脸的话对框进行修正


# -----------------------------#
#   mtcnn的第三段
#   精修框并获得五个点
# -----------------------------#
def create_Onet(weight_path):
    # 输入时把shape调整成48*48*3
    input = Input(shape=[48, 48, 3])
    # 48,48,3 -> 23,23,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # 23,23,32 -> 10,10,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)
    # 8,8,64 -> 4,4,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)
    # 4,4,64 -> 3,3,128
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu4')(x)
    # 3,3,128 -> 128,3,3
    # 调整维度
    x = Permute((3, 2, 1))(x)

    # 1152 -> 256
    # 铺平
    x = Flatten()(x)
    # 全连接
    x = Dense(256, name='conv5')(x)
    x = PReLU(name='prelu5')(x)

    # 鉴别
    # 256 -> 2 256 -> 4 256 -> 10
    # 再进行三个全连接,ONet还要输出5个人脸特征点的位置
    # 判断这张图片里面到底有没有人脸,置信度
    classifier = Dense(2, activation='softmax', name='conv6-1')(x)
    # 对框的修正方案(精修),每个框的调整方式
    bbox_regress = Dense(4, name='conv6-2')(x)
    # 5个人脸特征点的位置
    landmark_regress = Dense(10, name='conv6-3')(x)

    model = Model([input], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)

    return model


class mtcnn():
    def __init__(self):
        self.Pnet = create_Pnet('model_data/pnet.h5')
        self.Rnet = create_Rnet('model_data/rnet.h5')
        self.Onet = create_Onet('model_data/onet.h5')

    def detectFace(self, img, threshold):
        # -----------------------------#
        #   归一化
        # -----------------------------#
        copy_img = (img.copy() - 127.5) / 127.5
        # 获得图片的长和宽
        origin_h, origin_w, _ = copy_img.shape

        # -----------------------------#
        #   计算原始输入图像
        #   每一次缩放的比例
        #   得到图像金字塔
        #   目的是为了检测不同大小的人脸,因为一张图片中可能会有若干不同大小的人脸
        # -----------------------------#
        scales = utils.calculateScales(img)
        #   创建一个列表,存放输出结果
        out = []
        # -----------------------------#
        #   粗略计算人脸框
        #   将图像金字塔中的每一张图片通过PNet获得每一张图片的初步特征提取效果
        # -----------------------------#
        for scale in scales:
            # 对图像金字塔中的每一张图片都先计算长和宽
            hs = int(origin_h * scale)  # 从比较合适的初始大小开始,不断乘0.709的若干次方,直到12
            ws = int(origin_w * scale)
            # 把原图resize成缩放后的图片
            scale_img = cv2.resize(copy_img, (ws, hs))
            inputs = scale_img.reshape(1, *scale_img.shape)
            # 利用PNet进行预测
            ouput = self.Pnet.predict(inputs)
            out.append(ouput)
        # 初步特征提取效果还需要处理后才是原图上的坐标,接下来是初步特征提取效果的解码过程
        image_num = len(scales)
        rectangles = []
        # 对每一张缩放后的图片进行处理
        for i in range(image_num):
            # 取出每一个网格点上有人脸的概率
            cls_prob = out[i][0][0][:, :, 1]
            # 取出每一个网格点所对应的框的位置
            roi = out[i][1][0]
            # 取出每个缩放后图片的长宽
            # PNet输出的宽和高
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            # 解码过程
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h,
                                                threshold[0])
            rectangles.extend(rectangle)

        # for i in range(len(rectangles)):
        #     bbox = rectangles[i]
        #     crop_img = img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        #     if bbox[3]-bbox[1]>80:
        #         cv2.imshow("crop_img",crop_img)
        #         cv2.waitKey(0)
        # 进行非极大抑制
        rectangles = utils.NMS(rectangles, 0.7)
        # 一大堆的矩形框保存在rectangle参数里,有矩形框的位置和得分5个参数,矩形框作为RNet的输入
        if len(rectangles) == 0:
            return rectangles

        # -----------------------------#
        #   稍微精确计算人脸框
        #   Rnet部分
        # -----------------------------#
        # 把所有矩形框里面的图片截出来
        predict_24_batch = []
        for rectangle in rectangles:
            # 截下来
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            # resize到24*24上,因为RNet的输入要求是24
            scale_img = cv2.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)

        predict_24_batch = np.array(predict_24_batch)
        # 传入到RNet里面
        out = self.Rnet.predict(predict_24_batch)
        # 框里有没有人脸的可信度
        cls_prob = out[0]
        cls_prob = np.array(cls_prob)
        # 如何调整某一张图片对应的rectangle
        roi_prob = out[1]
        roi_prob = np.array(roi_prob)
        # 筛选出具有高可信度人脸,对rectangle进行调整
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])
        if len(rectangles) == 0:
            return rectangles

        # -----------------------------#
        #   计算人脸框
        #   onet部分
        # -----------------------------#
        predict_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            # resize到48*48上面
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)

        predict_batch = np.array(predict_batch)
        output = self.Onet.predict(predict_batch)
        # 得到3个输出,除去要输出每个框调整后
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]
        # 筛选
        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
        return rectangles

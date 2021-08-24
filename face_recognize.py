from net.mtcnn import mtcnn
from net.facenet import facenet
from net.emotioncnn import EmotionCNN
from statistics import mode
from keras.layers import Input
import utils.utils as utils
import numpy as np
import cv2
import os

# PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))


class face_rec():

    # 初始化函数，进行所有人脸的编码以及landmark的框选等
    def __init__(self):
        # 创建mtcnn
        self.mtcnn_model = mtcnn()
        # 阈值确定
        self.threshold = [0.5, 0.8, 0.9]
        # MobileNet载入facenet
        # 并且将检测到的人脸转化为128维的向量便于之后处理
        input_shape = [160,160,3]
        self.facenet_model = facenet(input_shape=input_shape)
        # 将模型保存到model_data文件夹中
        path = './model_data/ep009-loss0.214-val_loss0.419.h5'
        # 载入模型
        self.facenet_model.load_weights(path, by_name=True)


        # 对人脸进行编码

        # 把所有的人脸存储进一个列表
        faces = os.listdir("face_dataset")
        # 存储人脸编码（用于计算距离获得相似度）
        self.face_codes = []
        # 存储对应的姓名
        self.face_names = []

        # 遍历所有人脸，筛选出最高得分
        for face in faces:
            # 所有存储的人脸图片都以“姓名缩写.后缀”命名
            name = face.split(".")[0]
            # 读取图片，把颜色格式从BGR转化为RGB
            data_path = os.path.join("./face_dataset/", face)
            image = cv2.imread(data_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 用矩形框选出满足阈值条件的人脸的位置
            rectangles = self.mtcnn_model.detectFace(image, self.threshold)
            # 转化成正方形，便于传入facenet
            rectangles = utils.rect2square(np.array(rectangles))

            # 取最好的框
            rectangle = rectangles[0]
            # 五个标记点全部减去人脸左上角的坐标,得到人脸上五个特征点相对于人脸左上角的相对坐标,这些坐标全部除以原有人脸框的宽度,*160,因为facenet输入是160*160
            landmark = (np.reshape(rectangle[5:15], (5, 2)) -
                        np.array([int(rectangle[0]), int(rectangle[1])])) / (rectangle[3] - rectangle[1]) * 160

            # 选出上面选中的矩形框
            crop_image = image[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            # 由于facenet要传入一个160x160的图片，所以将其resize
            crop_image = cv2.resize(crop_image, (160, 160))
            # 通过landmark实现人脸对齐
            new_image, temp = utils.Alignment_1(crop_image, landmark)

            # cv2.imshow("img",new_image)
            # cv2.waitKey(0)

            new_image = np.expand_dims(new_image, 0)

            # 对新图片计算编码，存储到face_codes中，
            face_code = utils.calc_128_vec(self.facenet_model, new_image)
            self.face_codes.append(face_code)
            self.face_names.append(name)

    # 人脸识别，与所有人脸进行匹配，选取得分最高的作为结果
    def recognize(self, draw):
        # 提取图像宽高
        height, width, temp = np.shape(draw)
        # 颜色转换
        draw_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # 检测人脸
        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)
        # 初始化后rectangles为空，说明没有框选到人脸
        if len(rectangles) == 0:
            return

        # 转化成正方形，并去掉超出图像的部分
        rectangles = utils.rect2square(np.array(rectangles, dtype=np.int32))
        rectangles[:, 0] = np.clip(rectangles[:, 0], 0, width)
        rectangles[:, 1] = np.clip(rectangles[:, 1], 0, height)
        rectangles[:, 2] = np.clip(rectangles[:, 2], 0, width)
        rectangles[:, 3] = np.clip(rectangles[:, 3], 0, height)

        # 对检测到的人脸进行编码
        face_codes = []
        for rectangle in rectangles:
            # 五个标记点全部减去人脸左上角的坐标,得到人脸上五个特征点相对于人脸左上角的相对坐标,这些坐标全部除以原有人脸框的宽度,*160,因为facenet输入是160*160
            landmark = (np.reshape(rectangle[5:15], (5, 2)) -
                        np.array([int(rectangle[0]), int(rectangle[1])])) / (rectangle[3] - rectangle[1]) * 160

            # 选出上面选中的矩形框
            crop_image = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            # 通过landmark实现人脸对齐
            new_image, temp = utils.Alignment_1(crop_image, landmark)
            # 由于facenet要传入一个160x160的图片，所以将其resize
            emo_image = cv2.resize(new_image, (64, 64))
            new_image = cv2.resize(new_image, (160, 160))
            new_image = np.expand_dims(new_image, 0)

            my_emo_rec = emotion_rec()
            emotion_text, emotion_rate = my_emo_rec.recognize(emo_image)

            # 对图片计算编码，存储到face_codes中，
            face_code = utils.calc_128_vec(self.facenet_model, new_image)
            face_codes.append(face_code)

        # 检测到的人脸对应姓名列表
        face_names = []
        # 开始进行人脸比对
        for face_code in face_codes:
            # 取出一张脸并与数据库中所有的人脸进行对比，计算得分
            scores = utils.compare_faces(self.face_codes, face_code)
            name = "Unknown"
            # 找出距离最近的人脸
            face_distances = utils.face_distance(self.face_codes, face_code)
            # 取出这个最近人脸的评分对应的序号
            best_position = np.argmin(face_distances)

            # 相似度很高，则判定为比对成功
            if scores[best_position] < 0.85:
                # 更新name
                name = self.face_names[best_position]
            face_names.append(name)

            # 对人脸库中的所有人脸进行人为评分
            face_values = [30, 100, 70, 60, 100]
            # 找到了比对成功的人脸
            if name != "Unknown":
                print('/*********************************/')
                print('识别成功！欢迎' + name + '！')
                print('angry:{:.2f}%  disgust:{:.2f}%  fear:{:.2f}%  \nhappy:{:.2f}%  sad:{:.2f}%  surprise:{:.2f}%  neutral:{:.2f}%'.format(emotion_rate[0], emotion_rate[1],
                                                                                                                                           emotion_rate[2], emotion_rate[3],
                                                                                                                                           emotion_rate[4], emotion_rate[5],
                                                                                                                                           emotion_rate[6]))

            else:
                print('/*********************************/')
                print('抱歉，您的人脸数据不在我们的人脸库中。识别失败！')

        # 前四个数字的切片是矩形框的长宽
        rectangles = rectangles[:, 0:4]

        # 生成矩形框
        for (left, top, right, bottom), name in zip(rectangles, face_names):
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = name + " " + emotion_text
            cv2.putText(draw, name, (left, bottom - 15),
                        font, 0.75, (173, 255, 47), 2)
        return draw

class emotion_rec():
    def __init__(self):
        input_shape = [64, 64, 1]
        inputs = Input(shape=input_shape, name="emotioncnn_input")
        self.emotion_model = EmotionCNN(inputs=inputs)
        #加载参数
        self.emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
        #加载模型
        emotion_model_path = './model_data/emotion_model_wights.h5'
        self.emotion_model.load_weights(emotion_model_path)

    def preprocess_input(self, x):
        x = x.astype('float32')
        x = x / 255.0
        x = x - 0.5
        x = x * 2.0
        return x

    #情绪识别
    def recognize(self, face):
        emotion_window = []
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_face = self.preprocess_input(gray_face)

        # cv2.imshow("copy_img",cv2.resize(gray_face, (128,128)))
        # cv2.waitKey(0)

        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = self.emotion_model.predict(gray_face)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = self.emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > 10:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            pass
        emotion_prediction = emotion_prediction[0] / sum(emotion_prediction[0]) * 100

        return emotion_mode, emotion_prediction

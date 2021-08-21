import numpy as np
import cv2
import math

# -----------------------------#
#   计算原始输入图像
#   每一次缩放的比例
# -----------------------------#
def calculateScales(img):
    # 副本
    copy_img = img.copy()

    pr_scale = 1.0
    # copy_img.shape是读入图片后的一个元组
    # 分解元组并用h,w获得了图片的高,宽
    h, w, _ = copy_img.shape

    # 图片太大,要进行缩小
    if min(w, h) > 500:
        pr_scale = 500.0 / min(h, w)  # 是一个小于1的数
        w = int(w * pr_scale)
        h = int(h * pr_scale)
    # 图片太小,要进行放大
    elif max(w, h) < 500:
        pr_scale = 500.0 / max(h, w)  # 是一个大于1的数
        w = int(w * pr_scale)
        h = int(h * pr_scale)
    # 图片不大不小的话,pr_scale就是1

    scales = []
    factor = 0.709
    factor_count = 0
    minl = min(h, w)  # 经过放缩后高和宽中的较小值
    # 循环条件是大于等于12,缩放后图片长和宽的最小值>=12
    while minl >= 12:
        scales.append(pr_scale * pow(factor, factor_count))  # 0.709的若干次方
        minl *= factor
        factor_count += 1
    # 最终得到图像金字塔(记录了放缩的尺度,pr_scale,pr_scale*0.709的若干次方)
    return scales


# -------------------------------------#
#   对pnet处理后的结果进行处理
# -------------------------------------#
def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):
    # 翻转,第零维和第一维的翻转;第零维和第二维的翻转
    cls_prob = np.swapaxes(cls_prob, 0, 1)
    roi = np.swapaxes(roi, 0, 2)
    # PNet会对所求的图片进行压缩,而stride就是求压缩的比例
    stride = 0
    # stride略等于2
    if out_side != 1:
        stride = float(2 * out_side - 1) / (out_side - 1)
    # 网格有人脸的概率大于门限
    (x, y) = np.where(cls_prob >= threshold)
    # 翻转x,y
    boundingbox = np.array([x, y]).T
    # 找到对应在原图中的位置
    # mtcnn初步筛选人脸框的核心
    # 求左上角的一个点
    bb1 = np.fix((stride * (boundingbox) + 0) * scale)
    # 求右下角的点
    bb2 = np.fix((stride * (boundingbox) + 11) * scale)
    # 下面的代码是绘制bb1和bb2
    # plt.scatter(bb1[:,0],bb1[:,1],linewidths=1)
    # plt.scatter(bb2[:,0],bb2[:,1],linewidths=1,c='r')
    # plt.show()
    boundingbox = np.concatenate((bb1, bb2), axis=1)
    # 调整左上角和右下角两个基准点
    # dx1 dx2是左上角网格点的偏移坐标
    dx1 = roi[0][x, y]
    dx2 = roi[1][x, y]
    # dx3 dx4是右下角网格点的偏移坐标
    dx3 = roi[2][x, y]
    dx4 = roi[3][x, y]
    score = np.array([cls_prob[x, y]]).T
    offset = np.array([dx1, dx2, dx3, dx4]).T

    boundingbox = boundingbox + offset * 12.0 * scale

    rectangles = np.concatenate((boundingbox, score), axis=1)

    # 将长方形调整为正方形
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        # 把左上角的点和零对比
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        # 右下角的点和宽和高进行对比,不能超出原图片的宽和高
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    # 进行非极大抑制,目的是把得分高的框筛选出来,把其他和得分高的框重合率比较高但得分较低的框剔除
    return NMS(pick, 0.3)


# -------------------------------------#
#   对pnet处理后的结果进行处理
# -------------------------------------#
def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):
    prob = cls_prob[:, 1]
    # 筛选出哪些图片里有人脸的置信度比较高
    pick = np.where(prob >= threshold)

    rectangles = np.array(rectangles)
    # 原始的人脸框
    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]

    sc = np.array([prob[pick]]).T
    # 对原始人脸框的参数调整,调整系数
    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    # 求出宽和高
    w = x2 - x1
    h = y2 - y1

    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T

    rectangles = np.concatenate((x1, y1, x2, y2, sc), axis=1)
    # 弄成正方形,方便ONet的处理
    rectangles = rect2square(rectangles)
    pick = []
    # 防止突破图像的边缘
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    return NMS(pick, 0.3)


# -------------------------------------#
#   对onet处理后的结果进行处理
# -------------------------------------#
def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):
    prob = cls_prob[:, 1]
    # 只筛选出置信度高于门限的框
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)

    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]

    sc = np.array([prob[pick]]).T
    # 筛选出调整方案
    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]
    # 计算每一个框的宽度和高度
    w = x2 - x1
    h = y2 - y1
    # pts是每一张图的五个点对应的位置
    pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
    pts1 = np.array([(h * pts[pick, 5] + y1)[0]]).T

    pts2 = np.array([(w * pts[pick, 1] + x1)[0]]).T
    pts3 = np.array([(h * pts[pick, 6] + y1)[0]]).T

    pts4 = np.array([(w * pts[pick, 2] + x1)[0]]).T
    pts5 = np.array([(h * pts[pick, 7] + y1)[0]]).T

    pts6 = np.array([(w * pts[pick, 3] + x1)[0]]).T
    pts7 = np.array([(h * pts[pick, 8] + y1)[0]]).T

    pts8 = np.array([(w * pts[pick, 4] + x1)[0]]).T
    pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T

    # 左上角的点的调整
    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    # 右下角的点的调整
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T

    rectangles = np.concatenate((x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9),
                                axis=1)
    # 左上角   右下角  得分   5个人脸特征点

    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, rectangles[i][4],
                         rectangles[i][5], rectangles[i][6], rectangles[i][7], rectangles[i][8], rectangles[i][9],
                         rectangles[i][10], rectangles[i][11], rectangles[i][12], rectangles[i][13], rectangles[i][14]])
    return NMS(pick, 0.3)


# -------------------------------------#
#   非极大抑制
# -------------------------------------#
def NMS(rectangles, threshold):
    if len(rectangles) == 0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    I = np.array(s.argsort())
    pick = []
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])  # I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o <= threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle


# 由于facenet需要的输入是正方形，因此该函数将长方形调整为正方形
def rect2square(rectangles):
    # 以下操作均在第二维进行，即对rectangle进行！

    # width存储每个通道宽度列表
    width = rectangles[:, 2] - rectangles[:, 0]
    # height存储每个通道长度列表
    height = rectangles[:, 3] - rectangles[:, 1]
    # width和height逐位进行比较，取其中大的那个，然后进行转置
    # 这里采取向外扩大是为了避免信息的损失
    larger = np.maximum(width, height).T

    # 如果width小，在原值基础上 - 0.5 * （height - width），就是向左移动；如果width大，就是原值
    rectangles[:, 0] = rectangles[:, 0] + width * 0.5 - larger * 0.5
    # 调整长度，原理与上一句代码相同
    rectangles[:, 1] = rectangles[:, 1] + height * 0.5 - larger * 0.5
    # 在0和1已经确定的基础上，确定2，3。将lager的每个数字重复两遍并转化为一维数组，加到2和3中
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([larger], 2, axis=0).T

    return rectangles


# 人脸对齐函数，找到人的面部特征
def Alignment_1(img, landmark):
    # shape[0]读取landmark第一维度的长度，分两种粉脸对齐的方式进行相似变换
    # landmark通过眼睛定位，把眼睛调成水平以带动整张图像（本程序未涉及）
    if landmark.shape[0] == 68:
        x = landmark[36, 0] - landmark[45, 0]
        y = landmark[36, 1] - landmark[45, 1]
    # landmark有五个标记点，分别减去左上角的点，求倾斜程度
    elif landmark.shape[0] == 5:
        x = landmark[0, 0] - landmark[1, 0]
        y = landmark[0, 1] - landmark[1, 1]

    # 角度为0，不需要进行相似变换
    if x == 0:
        angle = 0
    # 计算出旋转的角度
    else:
        angle = math.atan(y / x) * 180 / math.pi

    # 中心位置坐标
    center = (img.shape[1] // 2, img.shape[0] // 2)

    # 获得相似变换后的矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    # 用这个矩阵获得“正”的图像
    new_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

    # 将矩阵转化为一维数组
    rotation_matrix = np.array(rotation_matrix)
    # 准备计算新的landmark
    new_landmark = []
    # 遍历landmark的第一维度
    for i in range(landmark.shape[0]):
        pts = []
        # 计算出了经过相似变换后新的landmark
        pts.append(rotation_matrix[0, 0] * landmark[i, 0] + rotation_matrix[0, 1] * landmark[i, 1] + rotation_matrix[0, 2])
        pts.append(rotation_matrix[1, 0] * landmark[i, 0] + rotation_matrix[1, 1] * landmark[i, 1] + rotation_matrix[1, 2])
        new_landmark.append(pts)

    # 转化为一维数组
    new_landmark = np.array(new_landmark)

    return new_img, new_landmark


# 对landmark和标准landmark进行处理
def transformation(std_landmark, landmark):
    # 将标准landmark和landmark转化为float类型
    std_landmark = np.matrix(std_landmark).astype(np.float64)
    landmark = np.matrix(landmark).astype(np.float64)

    # 进行标准landmark和landmark的归一化
    c1 = np.mean(std_landmark, axis=0)
    c2 = np.mean(landmark, axis=0)
    std_landmark -= c1
    landmark -= c2

    # 进行标准landmark和landmark的归一化
    s1 = np.std(std_landmark)
    s2 = np.std(landmark)
    std_landmark /= s1
    landmark /= s2

    # 对该矩阵进行奇异值分解
    U, S, Vt = np.linalg.svd(std_landmark.T * landmark)
    R = (U * Vt).T

    # 数组的垂直叠加
    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])


# 第二种人脸对齐方式
def Alignment_2(img, std_landmark, landmark):
    # 送入Transformation函数进行计算
    trans_matrix = transformation(std_landmark, landmark)
    # 取出第二维的数
    trans_matrix = trans_matrix[:2]
    trans_matrix = cv2.invertAffineTransform(trans_matrix)
    # 生成了新的图片
    new_image = cv2.warpAffine(img, trans_matrix, (img.shape[1], img.shape[0]))
    # 转化为一维数组
    trans_matrix = np.array(trans_matrix)
    new_landmark = []
    # 遍历landmark第一维度
    for i in range(landmark.shape[0]):
        # 计算出了经过相似变换后新的landmark
        pts = []
        pts.append(trans_matrix[0, 0] * landmark[i, 0] + trans_matrix[0, 1] * landmark[i, 1] + trans_matrix[0, 2])
        pts.append(trans_matrix[1, 0] * landmark[i, 0] + trans_matrix[1, 1] * landmark[i, 1] + trans_matrix[1, 2])
        new_landmark.append(pts)

    # 转化为一维数组
    new_landmark = np.array(new_landmark)

    return new_image, new_landmark


# 高斯归一化，计算特征值（编码）之前的预处理
def pre_process(x):
    # 如果图片维度为4，RGB-A图像
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    # 如果图片维度为3，RGB图像
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    # 图片维度错误
    else:
        raise ValueError('Dimension should be 3 or 4')

    # 高斯归一化

    # 计算各个维度的平均值
    mean = np.mean(x, axis=axis, keepdims=True)
    # 将各个维度的平均值标准化（归一）
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj

    # 返回归一化后的矩阵
    return y


# l2标准化（向量中每个元素除以向量的L2范数）
def l2_normalize(x, axis=-1, epsilon=1e-10):
    return x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))


# 计算128特征值，即对人脸进行编码
def calc_128_vec(model, img):
    # 首先对图像进行预处理
    face_image = pre_process(img)
    # 返回样本属于每一个类别的概率，进行语义分段
    pre = model.predict(face_image)
    # 对图像进行l2标准化
    pre = l2_normalize(np.concatenate(pre))
    # 转化为size为128的一维数组，作为特征
    pre = np.reshape(pre, [128])

    return pre


# 计算人脸距离
def face_distance(face_encodings, face_to_compare):
    # 没有需要计算的
    if len(face_encodings) == 0:
        return np.empty(0)
    # 计算欧氏距离
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


# 比较人脸，将相似得分放入列表
def compare_faces(known_face_encodings, face_encoding_to_check):
    # 计算距离
    distance = face_distance(known_face_encodings, face_encoding_to_check)
    # 返回距离列表
    return list(distance)

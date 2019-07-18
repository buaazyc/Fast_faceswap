import dlib
import numpy as np
# 调用dlib库模型

# 用于人脸边框识别
detector = dlib.get_frontal_face_detector()
# 68个特征点识别
predictor = dlib.shape_predictor("conf/shape_predictor_68_face_landmarks.dat")
# 128维特征向量
descriptor = dlib.face_recognition_model_v1('conf/dlib_face_recognition_resnet_model_v1.dat')
print("调用dlib模型完成")


def get_landmarks(img, padding=0.2):
    """
    利用dlib库函数得到img中人脸的矩形边框和特征点坐标
    :param img: 人脸图片
    :param padding:
    :return: 矩形边框list和人脸特征点坐标list
    """
    # 利用detector模型检测人脸边框rects
    rects = detector(img, 1)
    rect_list = []
    landmark_list = []

    # 判断识别到的人脸个数
    if len(rects) > 5:
        print('[Warning]: Too much face detected...(more than 5)')
    elif 1 < len(rects) < 5:
        print('[Warning]: More than one face in picture(2~5)')
    elif len(rects) == 0:
        print('[Error]: No face detected...')
    elif len(rects) == 1:
        # print("检测到人脸的个数为1个")
        rects = [rects[0]]

    # 遍历所有人脸，添加对应特征点到列表中
    for item in rects:
        # 二维数组landmark_list的元素为[(x, y), ...]
        landmark_list.append([(p.x, p.y) for p in predictor(img, item).parts()])
        # 二维数组rect_list的元素为[(top, bottom, left, right), ...]
        edges = [item.top(), item.bottom(), item.left(), item.right()]
        rect_list.append(padding_edge(edges, img.shape, padding=padding))
    # print("检测到特征点是数量为{}".format(len(landmark_list)))
    # print("人脸方框的坐标位置为(上，下，左，右){}".format(rect_list[0]))
    return landmark_list, rect_list


def padding_edge(rectangle, shape, padding=0.2):
    """
    为边界增加padding，比例为padding=0.2，同时处理边界溢出的情况
    :param rectangle: 矩阵边框的上下左右 [top, bottom, left, right]
    :param shape: img.shape 图片的分辨率
    :param padding: 边界填充的比例
    :return: 填充后的矩形
    """
    # padding的高和宽
    height = (rectangle[1] - rectangle[0]) * padding
    length = (rectangle[3] - rectangle[2]) * padding

    # 填充矩形，同时边界处理
    rectangle[0] = rectangle[0] - height if rectangle[0] - height > 0 else 0
    rectangle[1] = rectangle[1] + height if rectangle[1] + height < shape[0] else shape[0]
    rectangle[2] = rectangle[2] - length if rectangle[2] - length > 0 else 0
    rectangle[3] = rectangle[3] + length if rectangle[3] + length < shape[1] else shape[1]

    return np.array(rectangle, dtype=np.int).tolist()


def get_face_descriptor(img):
    # 求脸的特征向量
    # 调用dlib库模型
    landmarks = get_landmarks(img)[0]
    face_descriptor = descriptor.compute_face_descriptor(img, landmarks)  # (128, 1)特征向量
    return face_descriptor


def compare_descriptor(descriptor1, descriptor2):
    # 计算两个人脸特征的相似度
    assert descriptor1.shape == descriptor2.shape == (128, 1)
    d1 = np.array(descriptor1)
    d2 = np.array(descriptor2)
    diff = (d1 - d2) ** 2
    res = sum(diff) ** 0.5
    print("脸的相似度距离为(0~1：越小越相似)：{}".format(res))
    return res

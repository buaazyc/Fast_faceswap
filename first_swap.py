import numpy as np
import cv2
from copy import copy
from use_dlib import get_landmarks

# 右眼在68个特征点的索引值
RIGHT_EYE_POINTS = list(range(36, 42))
# 左眼
LEFT_EYE_POINTS = list(range(42, 48))
# 用于计算脸部高斯模糊
COLOUR_CORRECT_BLUR_FRAC = 0.6


def first_swap(people_id, background_id):
    """
    初步的换脸：
        1.利用dlib找到特征点
        2.利用特征点将人脸对齐
        3.调整色调
        4.获得对应替换位置的mask
    :param people_id: 对应照片号
    :param background_id: 对应背景号
    :return: 
    """
    img1 = cv2.imread('swp/SWP_{}.jpg'.format(background_id)).astype(np.uint8)
    img2 = cv2.imread('input/{}.jpg'.format(people_id)).astype(np.uint8)

    # 第一步：获取img1和img2的关键部位
    #

    landmarks1, t_list1 = get_landmarks(img1)
    landmarks1 = np.mat(landmarks1[0])
    # img2

    landmarks2, t_list2 = get_landmarks(img2)
    landmarks2 = np.mat(landmarks2[0])

    # 第二步：TransferPoints函数，获得变换矩阵M
    # 全脸68个特征点用于计算
    align_points = list(range(68))
    m = transfer_points(landmarks1[align_points], landmarks2[align_points])

    # 第三步：
    # 获得img1和img2的脸颊所产生的边框
    # 面部是白色，背景是黑色
    img1mask = get_face_mask(img1.shape, np.array(landmarks1))
    img2mask = get_face_mask(img2.shape, np.array(landmarks2))
    # cv2.imshow('img1mask', img1mask)
    # cv2.waitKey(0)
    # cv2.imshow('img2mask', img2mask)
    # cv2.waitKey(0)

    # 对mask和拍摄的图片进行仿射变换，旋转、平移、缩放等等
    warped_mask = warp_img(img2mask, m, img1.shape)
    warped_mask = np.min([img1mask, warped_mask], axis=0)
    warped_img2 = warp_img(img2, m, img1.shape)
    # cv2.imshow('warped_mask', warped_mask)
    # cv2.waitKey(0)
    cv2.imwrite('temp/warped_img2.jpg', warped_img2.astype(np.uint8))

    # 第四步：
    # 调整脸部色调
    warped_corrected_img2 = modify_color(img1, warped_img2, landmarks1)
    cv2.imwrite('temp/warped_corrected_img2.jpg', warped_corrected_img2)
    # cv2.imshow("warped_corrected_img2", np.clip(warped_corrected_img2, 0, 255).astype(np.uint8))
    # cv2.waitKey(0)
    return warped_mask, img1, warped_corrected_img2


def transfer_points(points1, points2):
    """
    得到仿射变换矩阵
    :param points1:
    :param points2:
    :return:
    """
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    # 奇异值分解
    u, s, v_t = np.linalg.svd(points1.T * points2)
    r = (u * v_t).T
    return np.vstack([np.hstack(((s2 / s1) * r, c2.T - (s2 / s1) * r * c1.T)),
                      np.mat([0., 0., 1.])])


def warp_img(img, m, d_shape):
    """
    调用cv2.warpAffine函数对图片进行仿射变换
    :param img: 待仿射变换图片
    :param m: 仿射变换矩阵，只使用前两行，第三行是[0,0,1]
    :param d_shape: 输出图片的大小
    :return: 输出图片
    """
    output_img = np.zeros(d_shape, dtype=img.dtype)
    cv2.warpAffine(img,
                   m[:2],
                   (d_shape[1], d_shape[0]),
                   dst=output_img,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_img


def boundary_expend_and_shrink(landmarks, offset=0.97):
    """
    根据特征点中心，调整特征点位置
    :param landmarks:
    :param offset:
    :return:
    """
    x = np.average(landmarks[:, 0])
    y = np.average(landmarks[:, 1])
    new_landmarks = [(int((x + (p[0] - x) * offset)), int((y + (p[1] - y) * offset))) for p in landmarks]
    return new_landmarks


def get_face_mask(img_shape, landmarks, color=(1, 1, 1)):
    """
    获得面部区域，并进行填充
    :param img_shape: 图片分辨率
    :param landmarks: 所有的脸部特征点
    :param color: 填充颜色
    :return: 填充后的面部区域
    """
    # 整个图片的黑色背景
    a_img = np.zeros(img_shape, dtype=np.float64)

    # 不包括下巴，但是包含0和16两个点（为了加入眼镜）
    a = list(range(17, 68))
    a.append(0)
    a.append(16)
    a.append(8)

    # 获得凸包
    points = cv2.convexHull(np.array(landmarks[a]))
    # 给凸包填充颜色
    a_img = cv2.fillConvexPoly(copy(a_img), points, color=color)

    # 只换五官
    # organs = [MOUTH_POINTS + NOSE_POINTS,
    #           RIGHT_BROW_POINTS + RIGHT_EYE_POINTS,
    #           LEFT_BROW_POINTS + LEFT_EYE_POINTS]
    # for organ in organs:
    #     points = cv2.convexHull(np.array(landmarks[organ]))
    #     a_img = cv2.fillConvexPoly(copy(a_img), points, color=color)

    return a_img


def modify_color(img1, img2, landmarks1):
    """
    将img2的图像色调转换为img1的图像色调
    :param img1: 背景图片
    :param img2: 人脸照片
    :param landmarks1: 目标图像的关键点
    :return: 调整后的图片
    """
    # 左右眼睛的平均位置[[x y]]和[[x y]]
    mean1 = np.mean(landmarks1[LEFT_EYE_POINTS], axis=0)
    mean2 = np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0)
    diff = mean1 - mean2
    # print('mean1{},mean2{},diff{}'.format(mean1, mean2, diff))

    # 求范式，此处默认二范数，即 根号（x^2 + y^2 + z^2 + ...）
    # 即两眼之间的距离，80左右
    blur_amount = int(COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(diff))

    # 高斯核必须是奇数
    if blur_amount % 2 == 0:
        blur_amount += 1

    # 高斯滤波
    img1_blur = cv2.GaussianBlur(img1, (blur_amount, blur_amount), 0)
    img2_blur = cv2.GaussianBlur(img2, (blur_amount, blur_amount), 0)
    # 避免出现除0的错误
    img2_blur += (128 * (img2_blur <= 1.0)).astype(img2_blur.dtype)

    # cv2.imshow("img", img2 * img1_blur / img2_blur)
    # cv2.waitKey(0)
    # img2 * img1_blur / img2_blur
    # return img2.astype(np.float64)
    return img2.astype(np.float64) * img1_blur.astype(np.float64) / img2_blur.astype(np.float64)


def color_change(background, people, blur_amount=51):
    # 高斯滤波
    img1_blur = cv2.GaussianBlur(background, (blur_amount, blur_amount), 0)
    img2_blur = cv2.GaussianBlur(people, (blur_amount, blur_amount), 0)
    # 避免出现除0的错误
    img2_blur += (128 * (img2_blur <= 1.0)).astype(img2_blur.dtype)
    res = people.astype(np.float64) * img1_blur.astype(np.float64) / img2_blur.astype(np.float64)
    return np.clip(res, 0, 255).astype(np.uint8)
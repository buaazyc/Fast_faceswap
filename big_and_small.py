import numpy as np
import cv2
from use_dlib import get_landmarks


def get_small(img1, img2):
    """
    小一号的版本，用于输入到风格迁移网络
    :param img1: background
    :param img2: people
    :return: 小一号的版本
    """
    img1 = np.clip(img1, 0, 255).astype(np.uint8)
    img2 = np.clip(img2, 0, 255).astype(np.uint8)

    global t_list2

    # padding=0减少图片尺寸
    landmark_list1, t_list1 = get_landmarks(img1, padding=0)
    landmark_list2, t_list2 = get_landmarks(img2, padding=0)
    # edges = [top(), bottom(), left(), right()]
    assert len(t_list1) == len(t_list2) == 1
    t_list1 = t_list1[0]
    t_list2 = t_list2[0]

    # 抠出人脸小图
    small_img1 = img1[t_list1[0]:t_list1[1], t_list1[2]:t_list1[3], :]
    small_img2 = img2[t_list2[0]:t_list2[1], t_list2[2]:t_list2[3], :]

    # 显示结果
    #     cv2.rectangle(img1, (t_list1[2], t_list1[0]), (t_list1[3], t_list1[1]), (0, 0, 255), 2)
    #     cv2.rectangle(img2, (t_list2[2], t_list2[0]), (t_list2[3], t_list2[1]), (0, 0, 255), 2)
    #     img1 = img1[:,:,::-1]
    #     img2 = img2[:,:,::-1]
    #     import matplotlib.pyplot as plt
    #     plt.subplot(2, 2, 1)
    #     plt.imshow(img1)
    #     plt.subplot(2, 2, 2)
    #     plt.imshow(img2)

    small_img1 = small_img1[:, :, ::-1]
    small_img2 = small_img2[:, :, ::-1]
    #     plt.subplot(2, 2, 3)
    #     plt.imshow(small_img1)
    #     plt.subplot(2, 2, 4)
    #     plt.imshow(small_img2)
    #     plt.show()

    # 写文件
    cv2.imwrite('small/1.jpg', small_img1[:, :, ::-1])
    cv2.imwrite('small/2.jpg', small_img2[:, :, ::-1])


def get_big(big_img, small_img):
    """
    把小图放在大图对应的位置
    """
    big_img[t_list2[0]:t_list2[1], t_list2[2]:t_list2[3], :] = small_img
    return big_img

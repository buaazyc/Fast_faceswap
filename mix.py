import cv2
import numpy as np
BLUR_AMOUNT = 51
SWAP = 1


def img_mix(mask, img1, img2, blur_amount=BLUR_AMOUNT, swap=SWAP):
    """
    图片融合
    :param mask: 脸部mask
    :param img1: 模板底片
    :param img2: 拍摄人脸
    :param blur_amount: 高斯核大小
    :param swap:
    :return:
    """

    # mask：脸部纯白色，背景纯黑色
    mask = cv2.GaussianBlur(mask * 255.0, (blur_amount, blur_amount), 0) / 255.0
    mask[np.where(mask > 0.8)] = 1.0
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    # temp1：模板图片，背景保留，脸部去除
    temp1 = img1 * (1.0 - mask)
    # cv2.imshow('temp1', np.clip(temp1, 0, 255).astype(np.uint8))
    # cv2.waitKey(0)

    # temp2：拍摄照片，背景去除，脸部保留
    temp2 = img2 * mask
    # cv2.imshow('temp2', np.clip(temp2, 0, 255).astype(np.uint8))
    # cv2.waitKey(0)

    # temp3：模板照片，背景去除，脸部保留
    temp3 = img1 * mask
    # cv2.imshow('temp3', np.clip(temp3, 0, 255).astype(np.uint8))
    # cv2.waitKey(0)

    res = temp1 + temp2 * swap + temp3 * (1.0 - swap)
    res = np.clip(res, 0, 255).astype(np.uint8)
    return res

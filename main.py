from first_swap import first_swap, color_change
from big_and_small import get_small, get_big
from change_style import change_style
from mix import img_mix
import cv2
import time


def main(people_id, background_id):
    # 记录开始时间
    time_begin = time.time()

    # 初步换脸
    mask, background, people = first_swap(people_id, background_id)
    time_first = time.time()
    print("初步换脸消耗的时间为：{}s".format(time_first-time_begin))

    # 抠取小脸用于输入神经网络
    get_small(background, people)

    # 风格迁移神经网络
    small_people = 'small/2.jpg'
    small_background = 'small/1.jpg'
    res_people = change_style(target_image_path=small_people, style_reference_image_path=small_background)

    # 还原大脸
    people = get_big(people, res_people)

    # 混合
    res = img_mix(mask, background, people)

    # 输出结果
    cv2.imwrite('output/{}to{}.jpg'.format(people_id, background_id), res)
    time_end = time.time()
    print("换脸总共耗时为{}秒".format(time_end-time_begin))
    # cv2.imshow("", res)
    # cv2.waitKey(0)

    print('成功将{}号人脸，换到了{}背景上！'.format(people_id, background_id))


if __name__ == '__main__':
    main(people_id=4, background_id=11)

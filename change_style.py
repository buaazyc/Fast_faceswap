from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
import time
import cv2
import numpy as np

# 不显示TensorFlow警告
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def change_style(target_image_path, style_reference_image_path):
    width, height = load_img(target_image_path).size

    global img_height
    global img_width

    img_height = height
    img_width = width

    # 三个图片张量，分别是目标图片、风格图片、生成图片
    # 目标图片和风格图片为常亮
    # 生成图片为TensorFlow中的占位符，用于传入数据
    target_image = K.constant(preprocess_image(target_image_path))
    style_reference_image = K.constant(preprocess_image(style_reference_image_path))
    # combination_image = K.placeholder((1, img_height, img_width, 3))
    combination_image = K.constant(preprocess_image(target_image_path))

    # 将三张图片合并为一个批量
    input_tensor = K.concatenate([target_image,
                                  style_reference_image,
                                  combination_image], axis=0)

    assert input_tensor.shape == (3, img_height, img_width, 3)

    # 利用三张图片组成的批量作为输入构建VGG19，加载模型使用预训练的ImageNet.
    model = vgg19.VGG19(input_tensor=input_tensor,
                        weights='imagenet',
                        include_top=False)

    # 映射层的名字为字典
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    # 用于内容损失的层
    content_layers = [
                      'block5_conv1'
                      ]
    # 用于风格损失的层
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    # 用于纹理损失的层
    texture_layers = [
                      'block1_conv1',
                      'block2_conv1',
                      'block3_conv1',
                      'block4_conv1'
                      ]

    # 损失加权平均时的权重
    total_variation_weight = 1e-4
    style_weight = 1.
    content_weight = 0.025 / len(content_layers)
    texture_weight = 0.025 / len(texture_layers)

    # 初始化权值
    loss = K.variable(0.)

    # 计算纹理损失
    for texture_layer in texture_layers:
        layer_features = outputs_dict[texture_layer]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss = loss + texture_weight * content_loss(style_reference_features, combination_features)

    # 计算内容损失
    for content_layer in content_layers:
        layer_features = outputs_dict[content_layer]
        target_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss = loss + content_weight * content_loss(target_image_features, combination_features)

    # 计算风格损失
    for layer_num, layer_name in enumerate(style_layers):
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss = loss + (style_weight * (layer_num + 1) / len(style_layers)) * sl

    # 计算总变差损失
    loss = loss + total_variation_weight * total_variation_loss(combination_image)

    # 设置梯度下降过程
    grads = K.gradients(loss, combination_image)[0]

    fetch_loss_and_grads = K.function([combination_image], [loss, grads])

    evaluator = Evaluator(fetch_loss_and_grads)

    # 迭代次数
    iterations = 3

    # 导入目标图片
    x_img = preprocess_image(target_image_path)
    x_img = x_img.flatten()
    for i in range(iterations):
        print("开始第{}次迭代".format(i + 1))
        start_time = time.time()
        x_img, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                             x_img,
                                             fprime=evaluator.grads,
                                             maxfun=20)
        print('当前损失值为:', min_val)

        # 合成的图片
        res_img = de_process_image(x_img.copy().reshape((img_height, img_width, 3)))

        end_time = time.time()
        print("第{}次迭代总共花费{}s".format(i + 1, end_time - start_time))

        #     # 显示图片
        #     print("第{}次迭代后，输出的图片为：".format(i+1))
        #     plt.imshow(res_img)
        #     plt.show()

        if i == iterations - 1:
            # 写照片
            res_img = res_img[:, :, ::-1]
            f_name = 'temp/style_changed_small_face.jpg'
            cv2.imwrite(f_name, res_img)
            return res_img


# 读取图片，改变图片的形状
def preprocess_image(image_path):
    # 按照图片路径导入图片，并改变图片的形状
    img = load_img(image_path, target_size=(img_height, img_width))
    # 图片变成数组
    img = img_to_array(img)
    # 拓展数组的形状
    img = np.expand_dims(img, axis=0)
    assert img.shape == (1, img_height, img_width, 3)
    # 为导入vgg19进行预训练
    img = vgg19.preprocess_input(img)
    return img


# vgg19.preprocess_input的逆操作
# 将vgg19网络输出的结果显示
def de_process_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 图像由BGR格式转化为RGB格式
    x = x[:, :, ::-1]
    # 像素值范围改为0到255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 内容损失
# 保证目标图像和生成图像在VGG19卷积神经网络的顶层具有相似的结果
# 二范式
def content_loss(base, combination):
    return K.sum(K.square(combination - base))


# 格拉姆矩阵
def gram_matrix(x):
    # batch_flatten:样本扁平化
    # permute_dimensions:轴置换
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    # 求向量内积
    gram = K.dot(features, K.transpose(features))
    return gram


# 风格损失函数
def style_loss(style, combination):
    s = gram_matrix(style)
    c = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(s - c)) / (4. * (channels ** 2) * (size ** 2))


# 总变差损失
# 对生成图像的像素进行操作，避免结果过度像素化
# 促使图像具有空间连续性
def total_variation_loss(x):
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, 1:, :img_width - 1, :]
    )
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, :img_height - 1, 1:, :]
    )
    return K.sum(K.pow(a + b, 1.25))


class Evaluator(object):
    def __init__(self, fetch_loss_and_grads):
        self.loss_value = None
        self.grads_values = None
        self.fetch_loss_and_grads = fetch_loss_and_grads

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = self.fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grads_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grads_values)
        self.loss_value = None
        self.grads_values = None
        return grad_values


if __name__ == '__main__':
    target_image_ = 'small/2.jpg'
    style_reference_image_ = 'small/1.jpg'
    result = change_style(target_image_, style_reference_image_)

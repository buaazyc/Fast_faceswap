"""
可视化卷积神经网络中间层
"""
from keras.applications import vgg19
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from keras import models


""" 导入图片和vgg19模型 """
image_path = 'swp/SWP_15.jpg'
img = load_img(image_path, target_size=(150, 150))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
img = vgg19.preprocess_input(img)
input_tensor = K.constant(img)
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet',
                    include_top=False)

layer_outputs = [layer.output for layer in model.layers]
print(len(layer_outputs))
print(model.input)
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activation_model.summary()
activations = activation_model.predict(input_tensor, steps=1)

layer_names = []
for layer in model.layers:
    layer_names.append(layer.name)

images_per_raw = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]  # 特征图中的特征个数
    size = layer_activation.shape[1]  # 特征图形状为（1，size，size，n_features)
    n_cols = n_features // images_per_raw
    display_grid = np.zeros((size * n_cols, images_per_raw * size))

    for col in range(n_cols):
        for row in range(images_per_raw):
            channel_image = layer_activation[0, :, :, col * images_per_raw + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
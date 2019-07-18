# fast_faceswap
fast_faceswap use dlib and change_style_network(基于dlib和风格迁移网络的快速换脸）

在传统的dlib换脸基础上，增加了基于风格迁移网络的部分，可以使得融合更加逼真，光照和肤色转化更加自然  
轻量级换脸参考：http://matthewearl.github.io/2015/07/28/switching-eds-with-python/  
风格迁移网络论文：A neural algorithm of artistic style
![image](https://github.com/buaazyc/fast_faceswap/blob/master/output/4to20.jpg)

##  算法流程
### 初步换脸流程：
1. 利用dlib检测68个人脸特征点  
2. 利用特征点进行人脸对齐  
3. 获得对应人脸位置mask  
4. 调整色调  

### 使用风格迁移网络流程：
5. 将上边检测到的人脸单独拿出来  
6. 将人脸照片和模板照片一起输入到网络中  
7. 网络首先将两个照片输入到已经训练好的vgg19中，得到各层激活  
8. 利用人脸block5_conv1计算内容损失，利用模板block1_conv1,block2_conv1,block3_conv1,block4_conv1计算纹理损失  
9. 最小化损失函数，迭代若干次后得到迁移后的人脸照片  
10. 与之前的照片融合  

## 环境：
配置环境的过程可能比较复杂，会遇到各种各样的错误，可以查阅各种博客进行解决，注意版本号的对应
主要包含:  
keras  
numpy  
opencv  
tensorflow  
dlib  
我所使用的环境版本号为：Windows7 + Cuda9.0 + python3.6 + tensorflow-gpu1.12 + keras2.2.4  

## 使用方法:
将人脸图片放在input文件夹下，以"阿拉伯数字编号+.jpg"格式保存，模板图片放在swp文件夹下  
main.py的main函数中，修改people_id为人脸图片的编号，background_id修改为模板图片的编号  
运行即可  
初步换脸时间大约为4s  
风格迁移时间根据gpu计算能力和迭代次数而不定，默认迭代了3次，具体可根据情况修改change_style.py下iterations  

## 注意：
1. 目前只考虑了一张照片中出现单个人脸的情况，没有考虑多张脸  
2. 如果模板图片人脸过大，则可能因为gpu计算能力较低而出现错误，人脸图片没有影响  
3. 如果因为文件过大，而不必下载vgg19训练权重，在运行时下载，但可能因为网络问题下载缓慢  


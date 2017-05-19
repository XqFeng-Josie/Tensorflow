#coding=utf-8
#图像预处理
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#读取图像的原始数据
image_raw_data = tf.gfile.FastGFile("./test.jpeg",'r').read()

with tf.Session() as sess:
    #将图像使用jpeg的格式进行解码从而得到图像的三维矩阵
    #对应的还有tf.image.decode_png
    img_data = tf.image.decode_jpeg(image_raw_data)
#图像翻转处理
    flipped1 = tf.image.flip_up_down(img_data)
    flipped2 = tf.image.flip_left_right(img_data)
    transposed = tf.image.transpose_image(img_data)
    #随机翻转
    flipped3 = tf.image.random_flip_left_right(img_data)
    flipped4 = tf.image.random_flip_up_down(img_data)
    plt.subplot(2,3,1)
    plt.imshow(img_data.eval())
    plt.subplot(2,3,2)
    plt.imshow(flipped1.eval())
    plt.subplot(2,3,3)
    plt.imshow(flipped2.eval())
    plt.subplot(2,3,4)
    plt.imshow(transposed.eval())
    plt.subplot(2,3,5)
    plt.imshow(flipped3.eval())
    plt.subplot(2,3,6)
    plt.imshow(flipped4.eval())
    
#图像色彩调整

#调整图像的亮度
    adjusted1 = tf.image.adjust_brightness(img_data,-0.5)
    adjusted2 = tf.image.adjust_brightness(img_data,+0.5)
    #在-0.7~0.7之间调整亮度
    adjusted_bri = tf.image.random_brightness(img_data,0.7)
#调整图像的对比度
    adjusted3 = tf.image.adjust_contrast(img_data,-5)
    adjusted4 = tf.image.adjust_contrast(img_data,+5)
    #在1～7之间调整对比度，不可为负值
    adjusted_con = tf.image.random_contrast(img_data,1,7)
#调整图像的色相
    adjusted5 = tf.image.adjust_hue(img_data,0.1)
    adjusted6 = tf.image.adjust_hue(img_data,0.9)
    #在0～0.5之间调整色相
    adjusted_hue = tf.image.random_hue(img_data,0.5)
#调整图像的饱和度
    adjusted7 = tf.image.adjust_saturation(img_data,-5)
    adjusted8 = tf.image.adjust_saturation(img_data,+5)
    #在1和7之间调整饱和度,不可为负值
    adjusted_sat = tf.image.random_saturation(img_data,1,7)
    
#图像标准化过程，亮度均值为0,方差变为1
    adjusted = tf.image.per_image_standardization(img_data)
    #图像矩阵转化为实数矩阵
    img_data = tf.image.convert_image_dtype(img_data,dtype=tf.float32)
#处理标注框
    boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
    begin,size,bbox_for_draw = tf.image.sample_distorted_bounding_box(tf.shape(img_data),bounding_boxes=boxes)
    #将图像矩阵加一维
   # batched = tf.expand_dims(img_data,0)
    #result = tf.image.draw_bounding_boxes(batched,bbox_for_draw)
    distorted_image = tf.slice(img_data,begin,size)
    
    plt.subplot(4,3,1)
    plt.imshow(adjusted1.eval())
    plt.subplot(4,3,2)
    plt.imshow(adjusted2.eval())
    plt.subplot(4,3,3)
    plt.imshow(adjusted_bri.eval())
    plt.subplot(4,3,4)
    plt.imshow(adjusted3.eval())
    plt.subplot(4,3,5)
    plt.imshow(adjusted4.eval())
    plt.subplot(4,3,6)
    plt.imshow(adjusted_con.eval())
    plt.subplot(4,3,7)
    plt.imshow(adjusted5.eval())
    plt.subplot(4,3,8)
    plt.imshow(adjusted6.eval())
    plt.subplot(4,3,9)
    plt.imshow(adjusted_hue.eval())
    plt.subplot(4,3,10)
    plt.imshow(adjusted7.eval())
    plt.subplot(4,3,11)
    plt.imshow(adjusted8.eval())
    plt.subplot(4,3,12)
    plt.imshow(adjusted_sat.eval())
    plt.show()
    plt.imshow(adjusted.eval())
    plt.show()

    plt.imshow(distorted_image.eval())
    plt.show()
    
    
    
   
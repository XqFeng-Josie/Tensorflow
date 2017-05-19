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
     #将表示一张图像的三维矩阵重新按照jpeg格式编码并存入文件，注意不能先转化再存储
    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile("./output_image","wb") as f:
        f.write(encoded_image.eval())
    #print img_data.eval()
    
    plt.subplot(2, 2, 1)
    plt.imshow(img_data.eval())
    #将数据的类型转化为实数,便于计算
    img_data = tf.image.convert_image_dtype(img_data,dtype=tf.float32)
    #使用不同的方法对图像进行大小改变
    resized1 = tf.image.resize_images(img_data,[300,300],method=0)
    resized2 = tf.image.resize_images(img_data,[300,300],method=1)
    resized3 = tf.image.resize_images(img_data,[300,300],method=2)
    plt.subplot(2, 2, 2) 
    plt.imshow(resized1.eval())
    plt.subplot(2, 2, 3) 
    plt.imshow(resized2.eval())
    plt.subplot(2, 2, 4) 
    plt.imshow(resized3.eval())
    plt.show()
    #img_data = tf.image.convert_image_dtype(img_data,dtype=tf.int32)
    #对图像进行裁剪或者填充
    #原始图像大于调整的尺寸，将会从图像中部抽取，反之在周围填充0
    croped = tf.image.resize_image_with_crop_or_pad(img_data,200,200)
    padded = tf.image.resize_image_with_crop_or_pad(img_data,400,400)
    #按比例调整图像大小
    central_cropped = tf.image.central_crop(img_data,0.5)
    plt.subplot(2, 2, 1)
    plt.imshow(img_data.eval())
    plt.subplot(2, 2, 2)
    plt.imshow(central_cropped.eval())
    plt.subplot(2, 2, 3)
    plt.imshow(croped.eval())
    plt.subplot(2, 2, 4)
    plt.imshow(padded.eval())
    plt.show()
    #给定区域裁剪，参数必须合适，否则报错,参数,图像数据，平移x,平移y，目标x,目标y
    bound_crop = tf.image.crop_to_bounding_box(img_data,30,30,200,200)
    plt.imshow(bound_crop.eval())
    plt.show()
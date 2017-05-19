#coding=utf-8
#在MNIST 数据集上实现神经网络，在5-3的基础上将代码更加标准模块化
#包含一个隐层
#5种优化方案：激活函数，多层隐层，指数衰减的学习率，正则化损失，滑动平均模型
#加入了tensorflow的变量管理思想
#训练和测试分开
#+加入了模型持久化的思想

#mnist_inference.py
#包含神经网络的前向传播过程以及神经网络中的参数

import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


#生成权重变量，并加入L2正则化损失到losses集合里
def get_weight(shape,Lamada):
    weights = tf.get_variable("weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if Lamada!=None:
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(Lamada)(weights))
    return weights
#对神经网络进行前向计算，有两个版本，包含滑动平均以及不包含滑动平均
#使用了RELU激活函数实现了去线性化，函数支持传入计算参数平均的类，便于测试时使用滑动平均模型·
def inference(input_tensor,avg_class,reuse,lamada):
    with tf.variable_scope('layer1',reuse=reuse):
            weights1 = get_weight([INPUT_NODE,LAYER1_NODE],lamada)
            biases1 = tf.get_variable("bias",[LAYER1_NODE],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
    with tf.variable_scope('layer2',reuse=reuse):
        weights2 = get_weight([LAYER1_NODE,OUTPUT_NODE],lamada)                              
        biases2 = tf.get_variable("bias",[OUTPUT_NODE],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))  
    if avg_class == None:
            layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1) 
            layer2 =tf.matmul(layer1,weights2)+biases2
    else:
        #首先需要使用avg_class.average函数计算变量的滑动平均值，然后再计算相应的神经网络前向传播结果
            layer1 = tf.nn.relu(
            tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
            layer2 = tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)
    return layer2

#coding=utf-8
#在MNIST 数据集上实现类似leNet5卷积神经网络
#包含一个两个卷积层，两个池化层和两个全连接层
#5种优化方案：激活函数，指数衰减的学习率，正则化损失，dropout,滑动平均模型
#加入了tensorflow的变量管理思想
#加入了模型持久化的思想

#leNet5_eval.py
#包含卷积神经网络的测试过程

import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#加载mnist_inference.py和mnist_train.py中定义的常量和参数
import leNet5_inference
import leNet5_train


#每10秒加载一次最新的模型，并在测试集上测试最新模型的正确率 
EVAL_INTREVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
            #调整输入数据的格式，输入为一个四维矩阵
        x = tf.placeholder(tf.float32,shape=(leNet5_train.BATCH_SIZE,#第一维表示一个batch里样例个数
                                               leNet5_inference.IMAGE_SIZE,#第二，第三表示图片的尺寸
                                               leNet5_inference.IMAGE_SIZE,
                                               leNet5_inference.NUM_CHANNELS),name='x_input')#第四维表示图片的深度，对于RGB 格式的图片，深度为3
        y_ = tf.placeholder(tf.float32,shape=(None,leNet5_inference.OUTPUT_NODE),name='y_input')
        #定义测试集数据
        xv,yv = mnist.test.next_batch(leNet5_train.BATCH_SIZE)
        reshape_xv = np.reshape(xv,(leNet5_train.BATCH_SIZE,
                                        leNet5_inference.IMAGE_SIZE,
                                        leNet5_inference.IMAGE_SIZE,
                                        leNet5_inference.NUM_CHANNELS))
        test_feed = {x:reshape_xv,y_:yv}
        #调用训练好的模型进行计算前向传播的结果,因为测试时不关注正则化损失的值，因此不用计算正则化损失
        y = leNet5_inference.inference(x,False,False,None)
        #进行验证集上的准确率计算
        #判断两个张量的每一维是否相等，如果相等就返回True,否则返回False
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        #这个运算先将布尔型的数值转为实数型，然后计算平均值，平均值就是准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
        variable_averages = tf.train.ExponentialMovingAverage(leNet5_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        #+初始化tf持久化类
        saver = tf.train.Saver(variables_to_restore)
        while True :
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(leNet5_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    print global_step
                    accuracy_score = sess.run(accuracy,feed_dict=test_feed)
                    print "After %s training step(s),test accuracy using average model is %g "%(global_step,accuracy_score)
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTREVAL_SECS)
def main(argv=None):
    mnist = input_data.read_data_sets("../../MNIST_data/",one_hot=True)
    evaluate(mnist)
if __name__ == '__main__':
    tf.app.run()
        

#coding=utf-8
#在MNIST 数据集上实现神经网络，在5-3的基础上将代码更加标准模块化
#包含一个隐层
#5种优化方案：激活函数，多层隐层，指数衰减的学习率，正则化损失，滑动平均模型
#加入了tensorflow的变量管理思想
#训练和测试分开
#加入了模型持久化的思想

#mnist_eval.py
#包含神经网络的测试过程

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#加载mnist_inference.py和mnist_train.py中定义的常量和参数
import mnist_inference
import mnist_train

#每10秒加载一次最新的模型，并在测试集上测试最新模型的正确率 
EVAL_INTREVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,shape=(None,mnist_inference.INPUT_NODE),name='x_input')
        y_ = tf.placeholder(tf.float32,shape=(None,mnist_inference.OUTPUT_NODE),name='y_input')
        #定义测试集数据
        test_feed = {x:mnist.test.images,
                         y_:mnist.test.labels}
        #调用训练好的模型进行计算前向传播的结果,因为测试时不关注正则化损失的值，因此不用计算正则化损失
        y = mnist_inference.inference(x,avg_class=None,reuse=False,lamada=None)
        #进行验证集上的准确率计算
        #判断两个张量的每一维是否相等，如果相等就返回True,否则返回False
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        #这个运算先将布尔型的数值转为实数型，然后计算平均值，平均值就是准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        #+初始化tf持久化类
        saver = tf.train.Saver(variables_to_restore)
        while True :
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy,feed_dict=test_feed)
                    print "After %s training step(s),test accuracy using average model is %g "%(global_step,accuracy_score)
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTREVAL_SECS)
def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST_data/",one_hot=True)
    evaluate(mnist)
if __name__ == '__main__':
    tf.app.run()
        

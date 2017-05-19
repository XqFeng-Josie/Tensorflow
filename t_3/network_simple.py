#coding=utf-8
#简单神经网络实现二分类的问题

import tensorflow as tf
from numpy.random import RandomState

batch_size=8

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x = tf.placeholder(tf.float32,shape=(None,2),name='x_input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name ='y_input')

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#tf.clip_by_value()可以将计算的数值限制在一个范围内（1e-10~1.0）
#y_表示真实值，y表示预测值，定义的是交叉熵损失函数
#对于回归问题，最常用的损失函数是均方误差（MSE）mse = tf.reduce_mean(tf.square(y_-y))
cross_entropy = -tf.reduce_mean(
            y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
#多分类问题适合softmax+cross_entrpy
#cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(y,y_)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
dataset_size=128
X = rdm.rand(dataset_size,2)
Y = [[int(x1+x2<1)] for (x1,x2) in X]

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print sess.run(w1)
    print sess.run(w2)
    
    steps = 5000
    for i in range(steps):
        start = (i*batch_size)%dataset_size
        end = min(start+batch_size,dataset_size)
        
        sess.run(train_step,feed_dict = {x:X[start:end],y_:Y[start:end]})
        if i%1000==0:
            total_cross_entropy = sess.run(
                cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training_step(s) ,cross_entropy on all data is %g"%(i,total_cross_entropy))
    print sess.run(w1)
    print sess.run(w2)
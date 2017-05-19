#coding=utf-8
#简单神经网络实现自定义损失函数（利润最大化）
#不包含隐层

import tensorflow as tf
from numpy.random import RandomState

batch_size=8

w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))

x = tf.placeholder(tf.float32,shape=(None,2),name='x_input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name ='y_input')

y = tf.matmul(x,w1)

#定义预测多了和少了的成本
loss_less = 10
loss_more = 1
#自定义损失函数
loss = tf.reduce_sum(tf.select(tf.greater(y,y_),
                               (y-y_)*loss_more,(y_-y)*loss_less))

                              

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

rdm = RandomState(1)
dataset_size=128
X = rdm.rand(dataset_size,2)
#加入了一个噪音值，-0.05～0.05之间
Y = [[x1+x2+rdm.rand()/10.0-0.05] for (x1,x2) in X]

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print sess.run(w1)
    
    steps = 5000
    for i in range(steps):
        start = (i*batch_size)%dataset_size
        end = min(start+batch_size,dataset_size)
        
        sess.run(train_step,feed_dict = {x:X[start:end],y_:Y[start:end]})
        if i%1000==0:
            total_loss = sess.run(
                loss,feed_dict={x:X,y_:Y})
            print("After %d training_step(s) ,loss on all data is %g"%(i,total_loss))
    print sess.run(w1)
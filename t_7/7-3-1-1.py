#coding=utf-8
#tf里队列的使用

import tensorflow as tf
#声明一个先进先出的队列，指明有两个元素，类型是Int32
q = tf.FIFOQueue(2,'int32')
#声明一个随机队列，参数：容量，出对列后的最小值，类型
#q = tf.RandomShuffleQueue(2,1,'int32')
#初始化队列
init = q.enqueue_many(([10,0],))
x=q.dequeue()
y=x+1
q_inc = q.enqueue([y])
with tf.Session() as sess:
    init.run()
    for _ in range(5):
        #运行q_inc将执行数据出对列，加一，入队列的过程
        v,_= sess.run([x,q_inc])
        print v
#coding=utf-8
#使用coordinator协同线程的处理
import tensorflow as tf
import numpy as np
import threading
import time

#线程中运行的程序，这个程序每隔一秒4判断是否需要打印自己的ID
def MyLoop(coord,worker_id):
    #使用tf.Coordinator类提供的协同工具判断是否需要停止并打印自己的ID 
    while not coord.should_stop():
        #随机停止所有的线程
        if np.random.rand()<0.1:
            print "Stoping from id:%d\n"%worker_id
            #调用coord.request_stop()函数通知其他线程停止
            coord.request_stop()
        else:
            print "Working on id:%d\n"%(worker_id)
        time.sleep(1)

coord = tf.train.Coordinator()
#声明创建5个进程
threads = [threading.Thread(target=MyLoop,args=(coord,i,))for i in xrange(5)]
for t in threads:t.start()
coord.join(threads)
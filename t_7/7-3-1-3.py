#coding=utf-8
#使用tf.QueueRunner和tf.Coordinator管理多线程队列操作
import tensorflow as tf

#声明一个先进先出的队列
queue = tf.FIFOQueue(100,'float')
#定义队列的入队操作
enqueue_op=queue.enqueue([tf.random_normal([1])])
#使用tf.train.QueueRunner创建多个线程运行队列的入队操作
qr = tf.train.QueueRunner(queue,[enqueue_op]*5)
#将定义过的QueueRunner加入tf计算图上指定的集合
#tf.train.add_queue_runner函数没有指定集合则加入默认的集合tf.GraphKeys,QUEUE_RUNNERS
tf.train.add_queue_runner(qr)
#定义出队操作
out_tensor = queue.dequeue()
with tf.Session() as sess:
    #使用tf.train.Coordinator来协同启动的线程
    coord = tf.train.Coordinator()
    #使用tf.QueueRunner时，需要明确调用tf.train.start_queue_runners来启动所有的线程，
    #否则，会因为没有执行入队操作而调用出对操作时，程序会一直等待队列操作。
    #tf.train.start_queue_runners会默认执行tf.GraphKeys.QUEUE_RUNNERS集合里的所有QueueRunner
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    #获取队列中的取值
    for _ in range(3):print sess.run(out_tensor)[0]
   # 使用 Coordinator来停止所有线程
    coord.request_stop()
    coord.join(threads)
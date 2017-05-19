#coding=utf-8
import tensorflow as tf
import reader #自己实现的reader

#读取数据并打印长度及前100位数据。
#存放原始数据的路径
DATA_PATH = "../simple-examples/data"
train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
print len(train_data)
#print train_data[:100]
#将训练数据组织成batch大小为4、截断长度为5的数据组。并使用队列读取前3个batch。
# ptb_producer返回的为一个二维的tuple数据。
result = reader.ptb_producer(train_data, 4, 5)
# 通过队列依次读取batch。
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(3):
        x, y = sess.run(result)
        print "X%d: "%i, x
        print "Y%d: "%i, y
    coord.request_stop()
    coord.join(threads)
#coding=utf-8
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#输入可以使用常量，但是数据量太大的时候，这样会生成大量的计算图
#x = tf.constant([[0.7,0.9]])
x=tf.placeholder(tf.float32,shape=(3,2),name='input')
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

sess = tf.Session()

#一起初始化      
init_top = tf.initialize_all_variables()
sess.run(init_top)
#分步初始化
#sess.run(w1.initializer)
#sess.run(w2.initializer)
#print tf.assign(w2,w1,validate_shape=False)


#print sess.run(y)#当使用placeholder时，因为没有传入输入，因此会报错吗
print (sess.run(y,feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))
sess.close()                
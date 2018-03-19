import tensorflow as tf

v = tf.Variable(10,dtype=tf.float32,name="v")
ema = tf.train.ExponentialMovingAverage(0.99)
print (ema.variables_to_restore())

saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess,'./model/model.ckpt')
    print (sess.run(v))#输出原来模型里面的滑动平均模型的值
    print (sess.run(tf.get_default_graph().get_tensor_by_name("v:0")))
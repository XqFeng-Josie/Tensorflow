import tensorflow as tf

v = tf.Variable(3,dtype=tf.float32,name="v")
ema = tf.train.ExponentialMovingAverage(0.99)
print ema.variables_to_restore()

saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess,'./model/model.ckpt')
    print sess.run(v)
    #print sess.run(tf.get_default_graph().get_tensor_by_name("add:0"))
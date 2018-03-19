import tensorflow as tf

v = tf.Variable(2,dtype=tf.float32,name="v")
v2 = tf.Variable(2,dtype=tf.float32,name="v2")

for var in tf.all_variables():
    print (var.name)
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_average_op = ema.apply(tf.all_variables())
for var in tf.all_variables():
    print (var.name)
saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    sess.run(tf.assign(v,10))
    sess.run(maintain_average_op)
    saver.save(sess,'model/model.ckpt')
    print (sess.run([v,ema.average(v)]))
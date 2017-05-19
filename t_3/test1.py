import tensorflow as tf
 
g1 = tf.Graph()
with g1.as_default():
	v=tf.get_variable("v",initializer=tf.zeros_initializer(shape=[1]))
g2 = tf.Graph()
with g2.as_default():
	v=tf.get_variable("v",initializer=tf.ones_initializer(shape=[1]))
with tf.Session(graph=g1) as sess:
	tf.initialize_all_variables().run()
	with tf.variable_scope("",reuse=True):
		print(sess.run(tf.get_variable("v")))
with tf.Session(graph=g2) as sess:
	tf.initialize_all_variables().run()
	with tf.variable_scope("",reuse=True):
		print(sess.run(tf.get_variable("v")))

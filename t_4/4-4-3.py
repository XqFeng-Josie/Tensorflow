#coding=utf-8
#滑动平均模型的小程序
#滑动平均模型可以使得模型在测试数据上更加健壮
import tensorflow as tf

#定义一个变量用以计算滑动平均，变量的初始值为0,手动指定类型为float32，
#因为所有需要计算滑动平均的变量必须是实数型
v1 = tf.Variable(0,dtype=tf.float32)

#模拟神经网络迭代的轮数，动态控制衰减率
step = tf.Variable(0,trainable=False)
#定义一个滑动平均的类，初始化时给定衰减率为0.99和控制衰减率的变量
ema = tf.train.ExponentialMovingAverage(0.99,step)

#定义一个滑动平均的操作，这里需要给定一个列表，每次执行这个操作时，列表里的元素都会被更新
maintain_average_op = ema.apply([v1])

with tf.Session() as sess:
    #初始化所有变量
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    
    #获取滑动平均之后变量的取值
    print sess.run([v1,ema.average(v1)])
    
    #更新v1的值为5
    sess.run(tf.assign(v1,5))
    #更新v1的滑动平均值，衰减率为min{0.99,(1+step)/(10+step)=0.1}=0.1,
    #所以v1的滑动平均被更新为0.1*0+0.9*5=4.5
    sess.run(maintain_average_op)
    print sess.run([v1,ema.average(v1)])
    
    #更新迭代的轮数
    sess.run(tf.assign(step,10000))
    sess.run(tf.assign(v1,10))
    #这里的衰减率变成0.99
    #v1 = 0.99*4.5+0.01*10=4.555
    sess.run(maintain_average_op)
    print sess.run([v1,ema.average(v1)])
    
    #再次更新滑动平均值
    sess.run(maintain_average_op)
    print sess.run([v1,ema.average(v1)])
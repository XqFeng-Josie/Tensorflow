#coding=utf-8
#tensorflow中集合的运用:损失集合
#计算一个5层神经网络带L2正则化的损失函数
import tensorflow as tf
from numpy.random import RandomState

#获得一层神经网络边上的权重，并将这个权重的L2 正则化损失加入名称为'losses'的集合里
def get_weight(shape,lamada):
    #生成对应一层的权重变量
    var = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lamada)(var))
    return var

x = tf.placeholder(tf.float32,shape=(None,2),name='x_input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y_input')

batch_size = 8

#定义每层神经网络的节点个数
layer_dimension=[2,10,10,10,1]
#获取神经网络的层数
n_layers = len(layer_dimension)
#这个变量表示前向传播时最深层的节点，最开始的时候是输入层
cur_layer = x
#当前层的节点个数
in_dimension = layer_dimension[0]

#通过一个循环生成5层全连接的神经网络结构
for i in range(1,n_layers):
    #获取下一层节点的个数
    out_dimension = layer_dimension[i]
    #获取当前计算层的权重并加入了l2正则化损失
    weight = get_weight([in_dimension,out_dimension],0.001)
    #随机生成偏向
    bias = tf.Variable(tf.constant(0.1,shape = [out_dimension]))
    #计算前向传播节点，使用RELU激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer,weight)+bias)
    #进入下一层之前，更新下一层节点的输入节点数
    in_dimension = layer_dimension[i]

#计算模型数据的均值化损失加入损失集合
mse_loss = tf.reduce_mean(tf.square(y_-cur_layer))
tf.add_to_collection('losses',mse_loss)

#get_collection返回一个列表，列表是所有这个集合的所有元素
#在本例中，元素代表了其他的损失，加起来就得到了所有的损失
loss = tf.add_n(tf.get_collection('losses'))

global_step = tf.Variable(0)
#学习率的设置：指数衰减法，参数：初始参数，全局步骤，每训练100轮乘以衰减速度0,96(当staircase=True的时候)
learning_rate = tf.train.exponential_decay(0.1,global_step,100,0.96,staircase=True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

rdm = RandomState(1)
dataset_size=128
X = rdm.rand(dataset_size,2)
#加入了一个噪音值，-0.05～0.05之间
Y = [[x1+x2+rdm.rand()/10.0-0.05] for (x1,x2) in X]

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    #print sess.run(w1)
    
    steps = 5000
    for i in range(steps):
        start = (i*batch_size)%dataset_size
        end = min(start+batch_size,dataset_size)
        
        sess.run(train_step,feed_dict = {x:X[start:end],y_:Y[start:end]})
        if i%1000==0:
            total_loss = sess.run(
                loss,feed_dict={x:X,y_:Y})
            print("After %d training_step(s) ,loss on all data is %g"%(i,total_loss))
    #print sess.run(w1)

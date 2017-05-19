#coding=utf-8
#在MNIST 数据集上实现神经网络
#包含一个隐层
#5种优化方案：激活函数，多层隐层，指数衰减的学习率，正则化损失，滑动平均模型
#+加入了tensorflow的变量管理思想

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
BATCH_SIZE = 100

#基础的学习率，使用指数衰减设置学习率
LEARNING_RATE_BASE = 0.8
#学习率的初始衰减率
LEARNING_RATE_DECAY = 0.99
#正则化损失的系数
LAMADA = 0.0001
#训练轮数
TRAINING_STEPS = 30000
#滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99

#生成权重变量，并加入L2正则化损失到losses集合里
def get_weight(shape,Lamada):
    weights = tf.get_variable("weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if Lamada!=None:
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(Lamada)(weights))
    return weights
#对神经网络进行前向计算，有两个版本，包含滑动平均以及不包含滑动平均
#使用了RELU激活函数实现了去线性化，函数支持传入计算参数平均的类，便于测试时使用滑动平均模型·
def inference(input_tensor,avg_class,reuse):
    with tf.variable_scope('layer1',reuse=reuse):
            weights1 = get_weight([INPUT_NODE,LAYER1_NODE],LAMADA)
            biases1 = tf.get_variable("bias",[LAYER1_NODE],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
    with tf.variable_scope('layer2',reuse=reuse):
        weights2 = get_weight([LAYER1_NODE,OUTPUT_NODE],LAMADA)                              
        biases2 = tf.get_variable("bias",[OUTPUT_NODE],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))  
    if avg_class == None:
            layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1) 
            layer2 =tf.matmul(layer1,weights2)+biases2
    else:
        #首先需要使用avg_class.average函数计算变量的滑动平均值，然后再计算相应的神经网络前向传播结果
            layer1 = tf.nn.relu(
            tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
            layer2 = tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)
    return layer2

#训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32,shape=(None,INPUT_NODE),name='x_input')
    y_ = tf.placeholder(tf.float32,shape=(None,OUTPUT_NODE),name='y_input')
    
    
    #计算神经网络的前向传播结果，注意滑动平均的类函数为None
    y = inference(x,avg_class=None,reuse=False)
    #定义存储模型训练轮数的变量，并指明为不可训练的参数
    global_step = tf.Variable(0,trainable=False)
    #初始化滑动平均的函数类，加入训练轮数的变量可以加快需年早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    #对神经网络里所有可训练参数（列表）应用滑动平均模型，每次进行这个操作，列表里的元素都会得到更新
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    #计算使用了滑动平均的网络前向传播结果，滑动平均不会改变变量本身的值，而是维护一个影子变量来记录其滑动平均值
    #因此当需要使用这个滑动平均值的时候，需要明确调用average函数
    average_y = inference(x,variable_averages,reuse=True)
    
    #当只有一个标准答案的时候，使用sprase_softmax_cross_entropy_with_logits计算损失，可以加速计算
    #参数：不包含softma层的前向传播结果，训练数据的正确答案
    #因为标准答案是一个长度为10的一维数组，而该函数需要提供一个正确答案的数字，
    #因此需要使用tf.argmax函数得到正确答案的对应类别编号
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,1),logits=y)
    #计算在当前batch里所有阳历的交叉熵平均值，并加入损失集合
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses',cross_entropy_mean)
    #get_collection返回一个列表，列表是所有这个集合的所有元素
    #在本例中，元素代表了其他部分的损失，加起来就得到了所有的损失
    loss = tf.add_n(tf.get_collection('losses'))
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,#基础的学习率，在此基础上进行递减
                                               global_step,#迭代的轮数
                                               mnist.train.num_examples/BATCH_SIZE,#所有的数据得到训练所需要的轮数
                                               LEARNING_RATE_DECAY)#学习率衰减速度
    #使用GradientDescentOptimizer()优化算法的损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    #在训练神经网络模型的时候，每过一边数据既需要通过反向传播更新网络的参数
    #又需要更新每一个参数的滑动平均值。为了一次完成多种操作，tensroflow提供了两种机制。
    #下面的两行程序和：train_op = tf.group(train_step,variables_average_op)等价
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')
    
    #进行验证集上的准确率计算，这时需要使用滑动平均模型
    #判断两个张量的每一维是否相等，如果相等就返回True,否则返回False
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    #这个运算先将布尔型的数值转为实数型，然后计算平均值，平均值就是准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    #初始化会话，并开始训练
    with tf.Session() as sess:
        #init_op = tf.initialize_all_variables()
        #sess.run(init_op)
        #初始化所有参数，同上面两句作用一致
        #tf.initialize_all_variables().run()#尽量用下面的初始化方式
	tf.global_variables_initializer().run()
        #准备验证数据，一般在神经网络的训练过程中会通过验证数据来判断大致停止的条件和评判训练的效果 
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        #准备测试数据，在实际中，这部分数据在训练时是不可见的，这个数据只是作为模型优劣的最后评价标准
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}
        #迭代的训练神经网络
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _,loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i%1000==0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))

                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print "After %d training step(s),validation accuracy using average model is %g "%(step,validate_acc)
                test_acc = sess.run(accuracy,feed_dict=test_feed)
                print("After %d training step(s) testing accuracy using average model is %g"%(step,test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    train(mnist)
if __name__ == '__main__':
    tf.app.run()
    

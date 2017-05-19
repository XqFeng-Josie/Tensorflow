#coding=utf-8
#预测正弦函数sin
#标准的RNN模型预测的是离散的数值，因此在程序中需要对连续的sin函数离散化，也就是
#在一定的时间间隔内对sin函数进行取样
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
from tensorflow.python.ops import array_ops as array_ops_
import matplotlib.pyplot as plt
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#加载TFLearm
learn = tf.contrib.learn

HIDDEN_SIZE = 30
NUM_LAYERS = 2

TIMESTEPS = 10 #RNN截断长度
TRAINING_STEPS = 3000#训练轮数
BATCH_SIZE = 32#batch大小

TRAINING_EXAMPLES = 10000#训练数据个数
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01#采样间隔

# 定义生成正弦数据的函数。
#定义使用sin函数一个TIMESTEPS长度的数据,去预测后面i+TIMESTEPS个点的函数值
def generate_data(seq):
    X = []
    y = []

    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

#定义lstm模型
def lstm_model(X, y):
    #使用多层的lstm结构
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS) 
    
    #计算前向传播结果，然后再reshape成[batch*timesteps,hidden_size]的形状
    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = tf.reshape(output, [-1, HIDDEN_SIZE])
    
    # 通过无激活函数的全联接层计算线性回归，并将数据压缩成一维数组的结构。
    predictions = tf.contrib.layers.fully_connected(output, 1, None)
    
    # 将predictions和labels调整统一的shape
    labels = tf.reshape(y, [-1])
    predictions=tf.reshape(predictions, [-1])
    #使用均方差损失函数计算损失
    loss = tf.losses.mean_squared_error(predictions, labels)
    #定义训练步骤
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(),
        optimizer="Adagrad", learning_rate=0.1)

    return predictions, loss, train_op

# 封装之前定义的lstm。
regressor = SKCompat(learn.Estimator(model_fn=lstm_model,model_dir="Model/model_2"))

# 生成数据。
test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP

#采样取值生成数据，用正弦函数生成训练和测试数据集合
#np.linspace()函数可以创建一个等差序列的数组，参数：起始值，终止值，数列的长度
train_X, train_y = generate_data(np.sin(np.linspace(
    0, test_start, TRAINING_EXAMPLES, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(
    test_start, test_end, TESTING_EXAMPLES, dtype=np.float32)))

# 训练模型
regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

# 计算预测值。
predicted = [[pred] for pred in regressor.predict(test_X)]

# 计算MSE（最大似然估计）作为评价指标
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print ("Mean Square Error is: %f" % rmse[0])
#绘图
plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(test_y, label='real_sin')
plt.legend([plot_predicted, plot_test],['predicted', 'real_sin'])
plt.show()
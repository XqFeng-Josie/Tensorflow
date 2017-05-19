#coding=utf-8
#使用TFlearn自定义模型的例子
#TFlearn集成在tf.contrib.learn里
#便于数据处理使用了sklearn工具包
from sklearn import model_selection
from sklearn import datasets
from sklearn import metrics
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
#解决编码问题
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#导入TFlearn
learn = tf.contrib.learn

#自定义模型，返回预测值，损失值以及训练步骤
def my_model(features, target):
    #将预测的目标转化成one-hot编码的形式，因为一共有三个类别，所以向量长度为3
    #经过转换后，类别分别表示为（1,0,0),(0,1,0),(0,0,1)
    target = tf.one_hot(target, 3, 1, 0)
    
    # 计算预测值及损失函数。
    #使用了一个全连接层，参数：输入，输出，激活函数
    logits = tf.contrib.layers.fully_connected(features, 3, tf.nn.softmax)
    loss = tf.losses.softmax_cross_entropy(target,logits)
    
    # 创建优化步骤。
    train_op = tf.contrib.layers.optimize_loss(
        loss,#损失函数
        tf.contrib.framework.get_global_step(),#获取训练步数并再训练时更新
        optimizer='Adam',#定义优化器
        learning_rate=0.01)#定义学习率
    return tf.arg_max(logits, 1), loss, train_op

iris = datasets.load_iris()
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=0)
#封装和训练模型，输出准确率
x_train, x_test = map(np.float32, [x_train, x_test])
classifier = SKCompat(learn.Estimator(model_fn=my_model, model_dir="Model/model_1"))
classifier.fit(x_train, y_train, steps=800)

y_predicted = [i for i in classifier.predict(x_test)]
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy: %.2f%%' % (score * 100))

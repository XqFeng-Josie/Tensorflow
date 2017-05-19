#coding=utf-8
#简单的RNN网络前向传播结构实现
import numpy as np

#定义输入以及初始状态，后面的状态都是动态计算
X=[1,2]
state = [0.0,0.0]

#分开定义参数，便于计算
w_cell_state = np.asarray([[0.1,0.2],[0.3,0.4]])
w_cell_input = np.asarray([0.5,0.6])
b_cell = np.asarray([0.1,-0.1])
w_output=np.asarray([[1.0],[2.0]])
b_output = np.asarray([0.1])

for i in range(len(X)):
    before_activation = np.dot(state,w_cell_state)+X[i]*w_cell_input+b_cell
    state = np.tanh(before_activation)
    final_output = np.dot(state,w_output)+b_output
    print "before activation:",before_activation
    print "state",state
    print  "output:",final_output
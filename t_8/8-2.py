#coding=utf-8
#简单LSTM 结构的RNN 的前向传播过程实现
import tensorflow as tf

lstm_hidden_size=1
batch_size=21
num_steps=22

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
lstm=tf.nn.run_cell.BasicLSTMCell(lstm_hidden_size)
state = lstm.zero_state(batch_size,tf.float32)
loss=0.0
'''for i in range(num_steps):
    if i>0:
        tf.get_variable_scope().reuse_variables()
        lstm_output,state = lstm(current_input,state)
        final_output = fully_connected(lstm_output)
        loss+=calc_loss(final_output,expected_output)
        '''
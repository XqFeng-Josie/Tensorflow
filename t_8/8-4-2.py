#coding=utf-8
#使用循环神经网络实现语言模型，使用的数据集是PTB 文本数据集
import numpy as np
import tensorflow as tf
import reader

DATA_PATH = '../simple-examples/data'
#隐藏层的规模
HIDDEN_SIZE = 200
#深层循环神经网络LSTM 结构的层数
NUM_LAYERS = 2
#词典规模，加上语句结束标识符和稀有单词标识符总共一万个单词
VOCAB_SIZE = 10000

#学习率
LEARNING_RATE = 1.0
TRAIN_BATCH_SIZE = 20
#训练数据截断的长度
TRAIN_NUM_STEP = 35

#在测试时不需要使用截断，将数据看作一个超长的序列
EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
#使用训练数据的轮数
NUM_EPOCH = 2
#dropout的概率
KEEP_PROB = 0.5
#用于控制梯度膨胀的参数
MAX_GRAD_NORM = 5

#通过一个PTBModel类来描述模型，便于维护RNN的状态
class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        #记录使用Batch的大小以及截断的长度
        self.batch_size = batch_size
        self.num_steps = num_steps
        
        # 定义输入层。输入层的维度是batch_size*num_steps
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        #定义预期输出，和输入层一致
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        
        # 定义使用LSTM结构及训练时使用dropout的deepRNN
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*NUM_LAYERS)
        
        # 初始化最初的状态,也就是全零的向量
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        
        #将单词ID 转化为单词向量。因为一共有VOCAB_SIZE个单词，每个单词向量的维度为HIDDEN_SIZE，因此embedding的参数维度为VOCAB_SIZE×HIDDENZ_SIZE
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])
        #原本的batch_size*num_steps个单词ID 转为单词向量，转化后输入层维度是batch_size*num_steps*hidden_size
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        
        #只在训练时使用dropout
        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)

        # 定义输出列表，先将不同时候的LSTM 结构的输出收集起来，再通过一个全连接层得到最终的输出。
        outputs = []
        #state里存储不同batch中LSTM 的状态，将其初始化为0
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                #从输入数据中获取当前时刻的输入并传入到LSTM 结构
                cell_output, state = cell(inputs[:, time_step, :], state)
                #将当前输出加入输出队列
                outputs.append(cell_output)
        #将输出队列展开成[batch,hidden_size*num_steps]的形状，然后再reshape成[batch*num_steps,hidden_size]的形状
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
        #将从LSTM 中得到的输出再经过一个全连接层得到最后的预测结果，最终的预测结果
        #在每一个时刻上都是一个长度为VOCAB_SIZE的数组，经过softmax层之后表示下一个位置是不同单词的概率
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias
        
        # 定义交叉熵损失函数和平均损失。
        #sequence_loss_by_example函数计算一个序列的交叉熵的和
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],#预测的结果
            [tf.reshape(self.targets, [-1])],#期待的正确的答案，将[batch_size,num_steps]二维数组转为一维数组
            #损失的权重，权重都为1,也即是不同batch和不同时刻的重要程度一致
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        #计算得到每个batch的平均损失
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state
        
        # 只在训练模型时定义反向传播操作。
        if not is_training: return
        trainable_variables = tf.trainable_variables()

        # 控制梯度大小，定义优化方法和训练步骤。
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        #定义优化方案
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        #定义训练步骤
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
#使用给定的模型model在3数据data上运行train_op并返回在全部数据熵的perplexity值
def run_epoch(session, model, data, train_op, output_log, epoch_size):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    # 使用当前数据训练或者测试模型
    for step in range(epoch_size):
        x, y = session.run(data)
        cost, state, _ = session.run([model.cost, model.final_state, train_op],
                                        {model.input_data: x, model.targets: y, model.initial_state: state})
        total_costs += cost
        iters += model.num_steps
        #只在训练时输出日志
        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (step, np.exp(total_costs / iters)))
    return np.exp(total_costs / iters)
def main():
    #获取原始数据
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

    # 计算一个epoch需要训练的次数
    train_data_len = len(train_data)
    train_batch_len = train_data_len // TRAIN_BATCH_SIZE
    train_epoch_size = (train_batch_len - 1) // TRAIN_NUM_STEP

    valid_data_len = len(valid_data)
    valid_batch_len = valid_data_len // EVAL_BATCH_SIZE
    valid_epoch_size = (valid_batch_len - 1) // EVAL_NUM_STEP

    test_data_len = len(test_data)
    test_batch_len = test_data_len // EVAL_BATCH_SIZE
    test_epoch_size = (test_batch_len - 1) // EVAL_NUM_STEP
    #定义初始化参数
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    #定义训练用的RNN模型
    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
    #定义评测用的RNN模型
    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    # 训练模型。
    with tf.Session() as session:
        tf.global_variables_initializer().run()

        train_queue = reader.ptb_producer(train_data, train_model.batch_size, train_model.num_steps)
        eval_queue = reader.ptb_producer(valid_data, eval_model.batch_size, eval_model.num_steps)
        test_queue = reader.ptb_producer(test_data, eval_model.batch_size, eval_model.num_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            run_epoch(session, train_model, train_queue, train_model.train_op, True, train_epoch_size)

            valid_perplexity = run_epoch(session, eval_model, eval_queue, tf.no_op(), False, valid_epoch_size)
            print("Epoch: %d Validation Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, eval_model, test_queue, tf.no_op(), False, test_epoch_size)
        print("Test Perplexity: %.3f" % test_perplexity)

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
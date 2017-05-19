#coding=utf-8

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

#Inception-v3模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048
#在Inception-v3模型里代表瓶颈层结果的张量名称。在谷歌提供的Inception-v3模型里，这个张量的
#名称就是‘pool_3/reshape:0’,在训练的模型时，可以通过tensor.name得到张量的名称
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
#图像输入张量所对应的名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

#模型目录
MODEL_DIR = './inception_dec_2015'
#模型文件名
MODEL_FILE= 'tensorflow_inception_graph.pb'
#因为一个训练数据会用到很多次，因此可以将原始图像通过Inception-v3模型计算得到的特征向量保存到文件里，免去重复的计算
CACHE_DIR = './cache'
#图片数据文件位置，在这个文件里，每个子文件夹下代表需要区分的类别，每个子文件夹里存放了对应类别的图片
INPUT_DATA = './flower_photos'
#验证的数据的百分比
VALIDATION_PERCENTAGE = 10
#测试的数据的百分比
TEST_PERCENTAGE = 10
#定义神经网络的设置
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

#将数据文件夹的图片按照训练，验证和测试数据分开
def create_image_lists(testing_percentage, validation_percentage):
    #得到的所有图片都存在result这个字典里，这个字典的key为类别的名称，value也是一个字典，字典里存储了所有图片的名称
    result = {}
    
    #获取当前目录下所有的子目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    #得到的第一个目录是当前目录，不予考虑
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        #获取当前目录下所有的有效图片文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        #返回文件名
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            #把目录和文件名合成一个路径
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            #返回所有匹配的文件路径列表，并加入列表
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue
        #通过目录名获取类别名称
        label_name = dir_name.lower()
        
        # 初始化当前类别的训练数据集，验证数据集以及测试数据集
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # 随机划分数据
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        #将当前类别的数据放入结果字典
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
            }
    return result

#根据类别的名称，所属数据集和图片编号获取一张图片的地址
#image_lists:所有图片信息
#image_dir:根目录。注意存放图片数据的根目录和存放图片特征向量的根目录不一致
#label_name:类别的名称
#index:图片编号
#category：指明所处的数据集
def get_image_path(image_lists, image_dir, label_name, index, category):
    #获取给定类别的所有图片信息
    label_lists = image_lists[label_name]
    #根据所属数据集的名称获取集合里的全部图片信息
    category_list = label_lists[category]
    #获取图片的文件名
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    #最后得到的地址为根目录的地址加上类别的文件名+图片的名称
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path

#通过类别名称、所属数据集和图片编号获取经过Inception_v3模型处理之后的特征向量文件地址
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'
#使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    #将当前图片作为输入计算瓶颈张量的值。这个瓶颈张量的值就是这张图片新的特征向量
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    #经过卷积网络处理的结果是一个四维数组，需要将这个结果压缩乘一个特征向量（一维数组）
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values
#获取一张图片经过Inception-v3模型处理之后的特征向量。
#函数先试着寻找已经计算且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    #获取一张图片对应的特征向量文件的路径
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)
   #如果这个特征向量文件不存在，则通过Inception-v3模型来计算特征向量，并将结果存入文件
    if not os.path.exists(bottleneck_path):
        #获取原始的图片路径
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        #获取图片内容
        image_data = gfile.FastGFile(image_path, 'rb').read()
        #计算特征向量
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        #将计算得到的特征向量存入文件
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        #直接从文件里获取图片相应的特征向量
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values

#函数随机获取一个batch的图片作为训练数据
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        #随机一个类别和图片的编号加入当前的训练数据
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths
#获取全部的测试数据，在最后测试的时候需在所有的测试数据上计算正确率
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    #枚举所有的类别和每个类别中的测试图片
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            #通过Inception-v3计算图片对应的特征向量，并加入最终数据的列表
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category,jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

def main():
    #读取所有图片
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    
    # 读取已经训练好的Inception-v3模型。
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    #加载读取的Inception-v3模型，并返回数据输入所对应的张量以及计算瓶颈层结果所对应的张量
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    # 定义新的神经网络输入，这个输入就是新的图片经过Inception-v3模型前向传播到达瓶颈层的节点取值，也就是特征提取
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    #定义新的标准答案输入
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')
    
    # 定义一层全链接层来解决新的图片分类的问题，因为训练好的Inception-v3模型已经将原始的图片抽象为更加容易分类的特征向量，
    #因此不需要再训练那么复杂的神经网络来完成这个新的分类任务
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)
        
    # 定义交叉熵损失函数。
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    
    # 计算正确率。
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #tf.initialize_all_variables().run()
        # 训练过程。
        for i in range(STEPS):
            #每次获取一个batch的训练数据
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
            sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
            #在验证数据上测试正确率
            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' %
                    (i, BATCH, validation_accuracy * 100))
            
        # 在最后的测试数据上测试正确率。
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(
            sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step, feed_dict={
            bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

if __name__ == '__main__':
    main()
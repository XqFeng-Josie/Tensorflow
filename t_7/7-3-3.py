#coding=utf-8
#读取TFRecord文件
#以及使用coordinator协同多线程处理，以及使用tf.train.string_input_producer表示输入文件队列
#+组合训练数据
#tf.train.batch和tf.train.shuffle_batch除了实现数据的组合，还实现了并行化处理数据的方法
#如果需要多个线程处理不同文件的样例时，可以使用tr.train.shuffle_batch_join函数。
#这个函数会平均分配文件以保证不同文件里的数据会被尽量平均的使用
import tensorflow as tf

#创建一个readeer来读取TFRecord文件里的样例
reader = tf.TFRecordReader()
#  match_filename_once函数获取文件列表，支持正则表达式匹配
files = tf.train.match_filenames_once("./output.tfrecords")
#创建一队列来维护输入文件列表,shuffle参数支持随机打乱文件列表的文件出队顺序
filename_queue = tf.train.string_input_producer(files,shuffle=False)
#也可以不使用上面的match_filenames_once函数，
#filename_queue = tf.train.string_input_producer(["./output.tfrecords"],shuffle=False
#从文件里读出一个样例，也可以使用read_up_to函数一次读取多个样例
_,serialized_example=reader.read(filename_queue)
#解析读入的一个样例，可以使用parse_example一次解析多个样例
features = tf.parse_single_example(
    serialized_example,
    #Tensorflow提供两种不同的属性解析方法，一种方法是tf.FixedLenFeature,这样的方法解析的结果是一个Tensor
    #另一种方法是tf.VarLenFeature,这种方法得到的解析结果为SparseTensor，用以处理稀疏数据
    #这里解析数据的格式需要和上面程序写入数据的格式一致
    features={
        'image_raw':tf.FixedLenFeature([],tf.string),
        'pixels':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64),
    })
#tf.decode_raw可以将字符串解析成图像对应的像素数组
images = tf.decode_raw(features['image_raw'],tf.uint8)
labels = tf.cast(features['label'],tf.int32)
pixels = tf.cast(features['pixels'],tf.int32)

#组合训练数据tf.train.batch和tf.train.shuffle_batch
#一个batch里样例的个数
batch_size=3
#组合样例的队列里最多可以存储的样例个数。队列如果太大，则会占用很多的内存资源，太少，那么出队操作可能会因为没有数据而被阻碍(block)
#从而导致训练效率降低，一般队列大小与batch的大小相关，如果设置了min_after_dequeue那么ca=min+3*bat
capacity = 1000+3*batch_size
#因为这里用的mnist数据的tfrecords格式文件，需要做一些处理，否则会报错
image = tf.reshape(images, [1, 784, 1])
#这里用的是shuffle_batch操作，会将顺序打乱
image_batch, label_batch = tf.train.shuffle_batch(
            [image, labels], batch_size=batch_size, num_threads=2,
            capacity=capacity,
            min_after_dequeue=100)#出队后最小剩余量，数量太小的话，随机打乱数据的作用不大。当
                                #出队函数被调用，但是队列里的元素不够的时候，出队操作将会等待更多元素入队才会完成

with tf.Session() as sess:
    #本例中没有声明变量，但是使用match_filenames_once函数需要初始化一些变量
    tf.global_variables_initializer().run()
    #启动多线程处理输入数据
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    #每次云霞可以读取TFRecord文件的一个样例，当所有的样例都读完之后，在此样例里程序会在重头读取。
    for i in range(2):
        image,label,pixel = sess.run([images,labels,pixels])
        #获取并打印组合之后的样例
        cur_image_batch,cur_label_batch = sess.run([image_batch,label_batch])
        print cur_image_batch,cur_label_batch
        #print pixel
    coord.request_stop()
    coord.join(threads)

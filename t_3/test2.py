#coding=utf-8
import tensorflow as tf

a = tf.constant([1.0,2.0],name='a')
b = tf.constant([3.0,4.0],name='b')
result = tf.add(a,b,name='add')
print result#返回张量结构
print tf.Session().run(result)
tf.Session().close()#显式关闭会话

#上下文管理会话资源
with tf.Session() as sess:
    print(sess.run(result))
    
    
sess = tf.Session()
#tf需要显式指定默认的会话，并通过tf.Tensor.eval()计算张量的取值
with sess.as_default():
    print(result.eval())
    #对应元素相乘，矩阵乘法式matmul
    print (a*b).eval()

sess = tf.Session()
#下面两句话有一样的功能
print(sess.run(result))
print(result.eval(session=sess))
                                                                                                                                                                                                                                                                        
#InteractiveSession()自动将生成的会话注册为默认的会话
sess = tf.InteractiveSession()
print(result.eval())
sess.close()

#通过ConfigProto对会话进行配置、、、、533----
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)
sess1.close()
sess2.close()

#声明一个2×3的矩阵变量，均值为0,标准差为2,另外还有random_gamma,truncated_normal,random_uniform
weights = tf.Variable(tf.random_normal([2,3],stddev=2,seed=1))

#除了使用随机数或者常量，tf还支持通过其他变量的值的初始化来初始化新的变量

w1 = tf.Variable(weights.initialized_value())
w2 = tf.Variable(weights.initialized_value()*2.0)

#产生全为0的数组
a = tf.Variable(tf.zeros([2,3],tf.int32))
#产生全为1的数组
b = tf.Variable(tf.ones([2,3],tf.int32))
#产生一个q全部为给定数字的数组
c = tf.Variable(tf.fill([2,3],9))
#产生一个给定值的常量
d = tf.Variable(tf.constant([2,3]))

#greater和select的应用
v1 = tf.constant([1.0,2.0,3.0,4.0])
v2 = tf.constant([4.0,1.0,2.0,5.0])

sess2 = tf.InteractiveSession()
print tf.greater(v1,v2).eval()
print tf.select(tf.greater(v1,v2),v1,v2).eval()
sess2.close()




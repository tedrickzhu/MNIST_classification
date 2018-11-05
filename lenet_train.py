#encoding = utf-8
__author__='zzy'
__date__ ='2018-08-08'

'''
input输入层
卷积核为3×3的conv2d卷积层
2×2的Maxpooling池化层
卷积核为3×3的conv2d卷积层
2×2的Maxpooling池化层
卷积核为3×3的conv2d卷积层,在LeNet论文中，输入矩阵为5X5X16，卷积核为5X5，和全连接层没有区别
625的full-connect全连接层
625to10 output输出层
'''

import tensorflow as tf
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 500

#初始化权重，输入维度，输出符合正太分布的变量
def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.01)
	return tf.Variable(initial)

#初始化偏置项，输入维度，输出初始值为0.0的偏置项
def bias_variable(shape):
	initial = tf.constant(0.0,shape=shape)
	return tf.Variable(initial)

'''
输入： X必须是四维张量，W为四维的卷积核，类型与X一致。
        strides，长度为四的一维整数类型数组，每一维度对应卷积核的中每一维的对应移动步数
        padding='SAME',仅适用于全尺寸操作，即输入数据与输出数据维度相同。
输出：X与W做卷积运算后的结果
'''
def conv2d(X,W):
	return tf.nn.conv2d(X,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(X):
    '''
    输入： X是一个四维张量，维度为[batch, height, width, channels]
            ksize,长度不小于4的整型数组，每一维上的值对应于输入数据张量中每一维的窗口对应值
            strides，指定滑动窗口在输入数据在输入数据张量每一维上的步长。
    '''
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

class CNN():

	def __init__(self, input_data_trX, input_data_trY, input_data_vaX,
	             input_data_vaY, input_data_teX, input_data_teY):
		self.w = None       #第一个卷积层的权重
		self.b = None       #第一个卷积层的偏置
		self.w2 = None      #第二个卷积层的权重
		self.b2 = None      #第二个卷积层的偏置
		self.w3 = None      #第三个卷积层的权重
		self.b3 = None      #第三个卷积层的偏置
		self.w4 = None      #全连接层中输入层到隐含层的权重
		self.b4 = None      #全连接层中输入层到隐含层的偏置
		self.w_o = None     #隐含层到输出层的权重
		self.b_o = None     #隐含层到输出层的偏置
		self.p_keep_conv = None     #卷积层中样本保持不变的比例
		self.p_keep_hidden = None   #全连接层中样本保持不变的比例
		self.trX = input_data_trX   #训练数据中的特征
		self.trY = input_data_trY   #训练数据中的标签
		self.vaX = input_data_vaX   #验证数据中的特征
		self.vaY = input_data_vaY   #验证数据中的标签
		self.teX = input_data_teX   #测试数据中的特征
		self.teY = input_data_teY   #测试数据中的标签

	def fit(self):
		'''
		调整输入数据placeholder的格式，输入X为一个四维矩阵
		第一维表示一个batch中样例的个数，初始为None
		第二维和第三维表示图片的尺寸，MNIST数据集中的图片为28X28
		第四维表示图片的深度，对于黑白图片，深度为1
		'''
		X = tf.placeholder(tf.float32, [None, 28, 28, 1])
		Y = tf.placeholder(tf.float32, [None, 10])

		# 第一层卷积核大小为3X3，输入一张图，输出32个feature map，在LeNet模型中，卷积核为5X5，
		# 但是对于28X28的图片来说，3X3的卷积核提取出的特征比5X5的要好，这样准确率也会提升。
		# 权重W的数组第三个维度是图片的通道数，第四个维度是卷积核的个数，也就是深度
		# 个数越多，网络越深，相应的计算复杂度也会提升，更加消耗计算资源。
		self.w = weight_variable([3, 3, 1, 32])
		self.b = bias_variable([32])
		# 第二层卷积核大小为3X3，输入32个feature map,输出64个feature map
		self.w2 = weight_variable([3, 3, 32, 64])
		self.b2 = bias_variable([64])
		# 第三个卷积核大小为3X3，输入64个feature map,输出128个feature map
		self.w3 = weight_variable([3, 3, 64, 128])
		self.b3 = bias_variable([128])
		# 全连接层FC 128 * 4 * 4 inputs, 625 outputs
		self.w4 = weight_variable([128 * 4 * 4, 625])
		self.b4 = bias_variable([625])
		# 全连接层FC 625 inputs, 10 outputs (labels)
		self.w_o = weight_variable([625, 10])
		self.b_o = bias_variable([10])
		self.p_keep_conv = tf.placeholder("float")  # 卷积层的dropout概率
		self.p_keep_hidden = tf.placeholder("float")  # 全连接层的dropout概率
		# 第一个卷积层
		l_c_l = tf.nn.relu(conv2d(X, self.w) + self.b)  # 1_c_1  shape=(?,28,28,32)
		l_p_l = max_pool_2x2(l_c_l)  # 1_p_1 shape(?, 14, 14, 32)
		# dropout:每个神经元有p_keep_conv的概率以1/p_keep_conv的比例进行归一化，
		# 有（1-p_keep_conv)的概率置为0
		l1 = tf.nn.dropout(l_p_l, self.p_keep_conv)

		# 第二个卷积层
		l_c_2 = tf.nn.relu(conv2d(l1, self.w2) + self.b2)  # 1_c_2 shape=(?, 14, 14, 64)
		l_p_2 = max_pool_2x2(l_c_2)  # l_p_2 shape=(?, 7, 7, 64)
		l2 = tf.nn.dropout(l_p_2, self.p_keep_conv)

		# 第三个卷积层
		l_c_3 = tf.nn.relu(conv2d(l2, self.w3) + self.b3)  # l_c_3 shape = (?, 7, 7,128)
		l_p_3 = max_pool_2x2(l_c_3)  # 1_p_3 shape=(?, 4, 4, 128)

		# 将所有的feature map合并成一个2048维向量
		l3 = tf.reshape(l_p_3, [-1, self.w4.get_shape().as_list()[0]])  # reshape to(?,2048)
		l3 = tf.nn.dropout(l3, self.p_keep_conv)

		# 后面两层为全连接层
		l4 = tf.nn.relu(tf.matmul(l3, self.w4) + self.b4)
		l4 = tf.nn.dropout(l4, self.p_keep_hidden)

		pyx = tf.matmul(l4, self.w_o) + self.b_o
		# 用交叉熵函数的平均值作为损失函数
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pyx, labels=Y))
		# RMSPro算法最小化目标函数
		train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
		predict_op = tf.argmax(pyx, 1)  # 返回每个样本的预测结果

		config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
		with tf.Session(config=config) as sess:
		    tf.global_variables_initializer().run()  # 初始化所有变量
		    for i in range(10):  # 训练10次
		        training_batch = zip(range(0, len(self.trX), batch_size), \
		                             range(batch_size, len(self.trX) + 1, batch_size))
		        for start, end in training_batch:  # 分批次进行训练
			        sess.run(train_op, feed_dict={X: self.trX[start:end], \
			                                      Y: self.trY[start:end], self.p_keep_conv: 0.8,
			                                      self.p_keep_hidden: 0.5})
		        if i % 3 == 0:
			        corr = np.mean(np.argmax(self.vaY, axis=1) == sess.run(predict_op, \
			                                                               feed_dict={X: self.vaX, Y: self.vaY,
			                                                                          self.p_keep_conv: 1.0, \
			                                                                          self.p_keep_hidden: 1.0}))
			        print("Accuracy at step %s on validation set:%s" % (i, corr))

		    # 最终在测试集上的输出
		    corr_te = np.mean(np.argmax(self.teY, axis=1) == sess.run(predict_op, \
		                                                              feed_dict={X: self.teX, Y: self.teY,
		                                                                         self.p_keep_conv: 1.0, \
		                                                                         self.p_keep_hidden: 1.0}))

		    print("Accuracy on test set : %s " % corr_te)
	        # 求测试集上的准确率时，上述代码在有GPU的电脑上运行时可能会报显存不足的错误，我尝试改成以下注释部分，
	# 又会出现警告，DeprecationWarning: elementwise == comparison failed; this will raise an error in the future.
	#                               self.p_keep_conv:1.0, self.p_keep_hidden:1.0}))
	# 结果显示为     Accuracy on test set : 0.0
	# 如果路过的朋友知道如何解决，还请多多指教。

	#            test_batch = zip(range(0, len(self.teX), batch_size),\
	#                                     range(batch_size, len(self.teX)+1, batch_size))
	#            for start, end in test_batch:
	#                corr_te = np.mean(np.argmax(self.teY, axis=1) == sess.run(predict_op,\
	#                              feed_dict={X: self.teX[start:end], Y: self.teY[start:end],
	#                                          self.p_keep_conv:1.0, self.p_keep_hidden:1.0}))
	#                print("Accuracy on test set : %s " % corr_te)


if __name__=="__main__":
    # 1.导入数据集
    start = time.clock() #计算开始时间
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #mnist.train.images是一个55000*784维的矩阵，mnist.train.labels是一个55000*10维的矩阵
    trX, trY, vaX, vaY, teX, teY = mnist.train.images,mnist.train.labels,\
                                    mnist.validation.images,mnist.validation.labels,\
                                    mnist.test.images, mnist.test.labels
    trX = trX.reshape(-1, 28, 28, 1)        #将每张图片用一个28X28的矩阵表示（55000，28，28，1)
    vaX = vaX.reshape(-1, 28, 28, 1)
    teX = teX.reshape(-1, 28, 28, 1)
    #2 .训练CNN模型
    cnn = CNN(trX, trY, vaX, vaY, teX, teY)
    cnn.fit()
    end = time.clock() #计算程序结束时间

    print("running time is", (end-start)/60,"min")





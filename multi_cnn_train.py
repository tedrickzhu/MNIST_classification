#encoding=utf-8
#author:Ethan
#software:Pycharm
#file:multi_cnn_train.py
#time:2018/11/5 上午11:21

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
    pass

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
    pass

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
    pass

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    pass


# mnist.train.images是一个55000*784维的矩阵，mnist.train.labels是一个55000*10维的矩阵
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#x，输入数据占位符
x = tf.placeholder("float", [None, 784])
# 训练数据的标准结果占位符
y_ = tf.placeholder("float", [None, 10])

# convolution layer 1
#卷积核为5x5，输入的维度为1，输出的维度为32，即，使用32个5x5的卷积核对图片做卷积，
# Padding为sanme，输出32个原图大小的特征图，然后用2x2的核池化，步长也为2，所以池化后特征图大小减半
w_conv1 = weight_variable([5, 5, 1, 32])
#每个输出维度添加一个偏置量
b_conv1 = bias_variable([32])
#将输入的图片转换成网络第一层需要的维度，长宽为28x28,一个通道，若为RGB格式，则为3个通道
x_image = tf.reshape(x, [-1, 28, 28, 1])
#卷积，然后用relu激活，特征图大小保持不变
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
#池化，特征图大小减半
h_pool1 = max_pool_2x2(h_conv1)
# convolution layer2
#卷积核为5x5，输入的维度为32，输出的维度为64，即，使用64个5x5的卷积核对图片做卷积，
# 每个卷积核和输入的32个特征图做卷积得到一个新的特征图
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# full connection layer
#现在图片减小到7x7，使用一个1024个神经元的全连接层
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
# dropout
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# output layer
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

#计算交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
#使用Adam优化器
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
#比较计算结果与真实结果是否相同
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#计算训练时的正确率
accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.initialize_all_variables())
'''
启动运行的两种方式，session.run(),options.eval()
使用options.eval()需要声明用的是哪个session，也就是用with ，或者是传入参数session=sess
accuracy.eval() 与sess.run()的作用是相同的，不同的是，eval必须返回数据，run可以返回，可以不返回
sess是一个session，accuracy 是一个options
'''
# with sess.as_default():
#     for i in range(20):
#         batch_x,batch_y = mnist.train.next_batch(100)
#         if i % 10 == 0:
#             train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
#             print('step %d ,training accuracy %g' % (i, train_accuracy))
#         train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
#
#     print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

for i in range(200):
    batch_x,batch_y = mnist.train.next_batch(100)
    if i % 10 == 0:
        train_accuracy = sess.run(accuracy,feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
        print('step %d ,training accuracy %g' % (i, train_accuracy))
    sess.run(train_step,feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})

print('test accuracy %g' % sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

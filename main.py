#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import        # 绝对导入
from __future__ import division                # 精确除法，/是精确除，//是取整除
from __future__ import print_function        # 打印函数

import os
import tensorflow as tf
import time
import numpy as np
import datetime


# 建立一个 cifar10_data 的类， 输入文件名队列，输出 labels 和images
class cifar10_data(object):

    def __init__(self, filename_queue):        # 类初始化

        # 根据上一篇文章介绍的文件格式，定义初始化参数
        self.height = 32
        self.width = 32
        self.depth = 3
        # label 一个字节
        self.label_bytes = 1
        # 图像 32*32*3 = 3072 字节
        self.image_bytes = self.height * self.width * self.depth
        # 读取的固定字节长度为 3072 + 1 = 3073 
        self.record_bytes = self.label_bytes + self.image_bytes
        self.label, self.image = self.read_cifar10(filename_queue)

    def read_cifar10(self, filename_queue):

        # 读取固定长度文件
        reader = tf.FixedLengthRecordReader(record_bytes = self.record_bytes)
        key, value = reader.read(filename_queue)
        record_bytes = tf.decode_raw(value, tf.uint8)
        # tf.slice(record_bytes, 起始位置， 长度)
        label = tf.cast(tf.slice(record_bytes, [0], [self.label_bytes]), tf.int32)
        # 从 label 起，切片 self.image_bytes = 3072 长度为图像
        image_raw = tf.slice(record_bytes, [self.label_bytes], [self.image_bytes])
        # 图片转化成 3*32*32
        image_raw = tf.reshape(image_raw, [self.depth, self.height, self.width])
        # 图片转化成 32*32*3
        image = tf.transpose(image_raw, (1,2,0))        
        image = tf.cast(image, tf.float32)
        return label, image


def inputs(data_dir, batch_size, train = True, name = 'input'):

    # 建议加上 tf.name_scope, 可以画出漂亮的流程图。
    with tf.name_scope(name):
        if train: 
            # 要读取的文件的名字
            filenames = [os.path.join(data_dir,'data_batch_%d.bin' % ii) 
                        for ii in range(1,6)]
            # 不存在该文件的时候报错
            for f in filenames:
                if not tf.gfile.Exists(f):
                    raise ValueError('Failed to find file: ' + f)
            #用文件名生成文件名队列
            filename_queue = tf.train.string_input_producer(filenames)
            # 送入 cifar10_data 类中
            read_input = cifar10_data(filename_queue)
            images = read_input.image
            # 图像白化操作，由于网络结构简单，不加这句正确率很低。
            # images = tf.image.per_image_whitening(images)
            labels = read_input.label
            # 生成 batch 队列，16 线程操作，容量 20192，min_after_dequeue 是
            # 离队操作后，队列中剩余的最少的元素，确保队列中一直有 min_after_dequeue
            # 以上元素，建议设置 capacity = min_after_dequeue + batch_size * 3
            num_preprocess_threads = 16
            image, label = tf.train.shuffle_batch(
                                    [images,labels], batch_size = batch_size, 
                                    num_threads = num_preprocess_threads, 
                                    min_after_dequeue = 20000, capacity = 20192)


            return image, tf.reshape(label, [batch_size])

        else:
            filenames = [os.path.join(data_dir,'test_batch.bin')]
            for f in filenames:
                if not tf.gfile.Exists(f):
                    raise ValueError('Failed to find file: ' + f)

            filename_queue = tf.train.string_input_producer(filenames)
            read_input = cifar10_data(filename_queue)
            images = read_input.image
            images = tf.image.per_image_whitening(images)
            labels = read_input.label
            num_preprocess_threads = 16
            image, label = tf.train.shuffle_batch(
                                    [images,labels], batch_size = batch_size, 
                                    num_threads = num_preprocess_threads, 
                                    min_after_dequeue = 20000, capacity = 20192)


            return image, tf.reshape(label, [batch_size])

def variable_on_cpu(name, shape, initializer = tf.constant_initializer(0.1)):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer = initializer, 
                              dtype = dtype)
    return var

 # 用 get_variable 在 CPU 上定义变量
def variables(name, shape, stddev): 
    dtype = tf.float32
    var = variable_on_cpu(name, shape, 
                          tf.truncated_normal_initializer(stddev = stddev, 
                                                          dtype = dtype))
    return var

# 定义网络结构
def inference(images):
    ''' 每次输入一个batch的 64 幅图像， 转化成 64*32*32*3 的四维张量，经过步长为 1，卷积核大小为 5*5 ，
    Feature maps 为64的卷积操作，变为 64*32*32*64 的四维张量，然后经过一个步长为 2 的 max_pool 的池化层，
    变成 64*16*16*64 大小的四维张量，再经过一次类似的卷积池化操作，
    变为 64*8*8*64 大小的4维张量，再经过两个全连接层，映射到 64*192 的二维张量，然后经过一个 sortmax 层，
    变为 64*10 的张量，最后和标签 label 做一个交叉熵的损失函数'''
    # 第一卷积层
    with tf.variable_scope('conv1') as scope:
        # 用 5*5 的卷积核，64 个 Feature maps
        weights = variables('weights', [5,5,3,64], 5e-2)
        # 卷积，步长为 1*1
        conv = tf.nn.conv2d(images, weights, [1,1,1,1], padding = 'SAME')
        biases = variable_on_cpu('biases', [64])
        # 加上偏置
        bias = tf.nn.bias_add(conv, biases)
        # 通过 ReLu 激活函数
        conv1 = tf.nn.relu(bias, name = scope.name)
        # 柱状图总结 conv1
        tf.summary.histogram(scope.name + '/activations', conv1)  
    with tf.variable_scope('pooling1_lrn') as scope:
        # 最大池化，3*3 的卷积核，2*2 的卷积
        pool1 = tf.nn.max_pool(conv1, ksize = [1,3,3,1], strides = [1,2,2,1],
                               padding = 'SAME', name='pool1')
        # 局部响应归一化
        norm1 = tf.nn.lrn(pool1, 4, bias = 1.0, alpha = 0.001/9.0, 
                          beta = 0.75, name = 'norm1')
    # 第二卷积层
    with tf.variable_scope('conv2') as scope:
        weights = variables('weights', [5,5,64,64], 5e-2)
        conv = tf.nn.conv2d(norm1, weights, [1,1,1,1], padding = 'SAME')
        biases = variable_on_cpu('biases', [64])
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name = scope.name)
        tf.summary.histogram(scope.name + '/activations', conv2)
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha = 0.001/9.0, 
                          beta = 0.75, name = 'norm1')        
        pool2 = tf.nn.max_pool(norm2, ksize = [1,3,3,1], strides = [1,2,2,1],
                               padding = 'SAME', name='pool1')

    with tf.variable_scope('local3') as scope:
        # 第一层全连接
        reshape = tf.reshape(pool2, [BATCH_SIZE,-1])
        dim = reshape.get_shape()[1].value
        weights = variables('weights', shape=[dim,384], stddev=0.004)
        biases = variable_on_cpu('biases', [384])
        # ReLu 激活函数
        local3 = tf.nn.relu(tf.matmul(reshape, weights)+biases, 
                            name = scope.name)
        # 柱状图总结 local3
        tf.summary.histogram(scope.name + '/activations', local3)

    with tf.variable_scope('local4') as scope:
        # 第二层全连接
        weights = variables('weights', shape=[384,192], stddev=0.004)
        biases = variable_on_cpu('biases', [192])
        local4 = tf.nn.relu(tf.matmul(local3, weights)+biases, 
                            name = scope.name)
        tf.summary.histogram(scope.name + '/activations', local4)

    with tf.variable_scope('softmax_linear') as scope:
        # softmax 层，实际上不是严格的 softmax ，真正的 softmax 在损失层
        weights = variables('weights', [192, 10], stddev=1/192.0)
        biases = variable_on_cpu('biases', [10])
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, 
                                name = scope.name)
        tf.summary.histogram(scope.name + '/activations', softmax_linear)

    return softmax_linear


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        labels = tf.cast(labels, tf.int64)
        # 交叉熵损失，至于为什么是这个函数，后面会说明。
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name = 'loss')
        tf.add_to_collection('losses', loss)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


BATCH_SIZE = 64      #设置batch的大小     每次输入一个batch的 64 幅图像
LEARNING_RATE = 0.1    #学习率
MAX_STEP = 50000    #循环次数

def train():
    # global_step
    global_step = tf.Variable(0, name = 'global_step', trainable=False)
    # cifar10 数据文件夹
    data_dir = './cifar-10-batches-bin/'
    # 训练时的日志logs文件，没有这个目录要先建一个
    train_dir = './logs/'
    # 加载 images，labels
    images, labels =inputs(data_dir, BATCH_SIZE)

    # 求 loss
    loss = losses(inference(images), labels)
    # 设置优化算法，这里用 SGD 随机梯度下降法，恒定学习率
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    # global_step 用来设置初始化
    train_op = optimizer.minimize(loss, global_step = global_step)
    # 保存操作
    saver = tf.train.Saver(tf.all_variables())
    # 汇总操作
    summary_op = tf.summary.merge_all()
    # 初始化方式是初始化所有变量
    init = tf.initialize_all_variables()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    # 占用 GPU 的 20% 资源
    #config.gpu_options.per_process_gpu_memory_fraction = 0.2
    # 设置会话模式，用 InteractiveSession 可交互的会话，逼格高
    sess = tf.InteractiveSession(config=config)
    # 运行初始化
    sess.run(init)

    # 设置多线程协调器
    coord = tf.train.Coordinator()
    # 开始 Queue Runners (队列运行器)
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    # 把汇总写进 train_dir，注意此处还没有运行
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

    # 开始训练过程
    for step in range(MAX_STEP):
        if coord.should_stop():
            break
        start_time = time.time()
        # 在会话中运行 loss
        _, loss_value = sess.run([train_op, loss])
        duration = time.time() - start_time
        # 确认收敛
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        if step % 30 == 0:
            # 本小节代码设置一些花哨的打印格式，可以不用管
            num_examples_per_step = BATCH_SIZE
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print (format_str % (datetime.datetime.now(), step, loss_value,
                                 examples_per_sec, sec_per_batch))

        if step % 100 == 0:
            # 运行汇总操作， 写入汇总
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)

        if step % 1000 == 0 or (step + 1) == MAX_STEP:
            # 保存当前的模型和权重到 train_dir，global_step 为当前的迭代次数
            checkpoint_path = os.path.join(train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

    coord.request_stop()
    coord.join(threads)

    sess.close()


def evaluate():

    data_dir = './cifar-10-batches-bin/'
    train_dir = './logs/cifar10_train/'

    images, labels =inputs(data_dir, BATCH_SIZE, train = False)

    logits = inference(images)
    saver = tf.train.Saver(tf.all_variables())

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.InteractiveSession(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    # 加载模型参数
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, os.path.join(train_dir, ckpt_name))
        print('Loading success, global_step is %s' % global_step)


    try:
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        true_count = 0
        step = 0
        while step < 157:
            if coord.should_stop():
                break
            predictions = sess.run(top_k_op)
            true_count += np.sum(predictions)
            step += 1

        precision = true_count / 10000
        print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
    except tf.errors.OutOfRangeError:
        coord.request_stop()
    finally:
        coord.request_stop()
        coord.join(threads)

    sess.close()
train()

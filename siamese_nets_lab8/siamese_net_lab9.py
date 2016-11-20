from __future__ import print_function

from pdb import set_trace as debugger
from imageloader import ImageLoader

import tensorflow as tf
import numpy as np

import os
import sys

batch_size = 128
resnet_units = 6

tf.reset_default_graph()

def global_avg_pool(in_var, name='global_pool'):
    assert name is not None, 'Op name should be specified'

    input_shape = in_var.get_shape()
    assert len(input_shape) == 4, 'Incoming Tensor shape must be 4-D'

    with tf.name_scope(name):
        inference = tf.reduce_mean(in_var, [1, 2])
    return inference


def max_pool(in_var, kernel_size=[1,2,2,1], strides=[1,1,1,1], 
            padding='SAME', name=None):
    assert name is not None, 'Op name should be specified'
    assert strides is not None, 'Strides should be specified when performing max pooling'
    # 
    with tf.name_scope(name):
        input_shape = in_var.get_shape()
        assert len(input_shape) == 4, 'Incoming Tensor shape must be 4-D'

        with tf.name_scope(name) as scope:
            inference = tf.nn.max_pool(in_var, kernel_size, strides, padding)

        return inference


def avg_pool_2d(in_var, kernel_size=[1,2,2,1], strides=None, 
                padding='SAME', name=None):
    assert name is not None, 'Op name should be specified'
    assert strides is not None, 'Strides should be specified when performing average pooling'
    # 
    with tf.name_scope(name) as scope:
        input_shape = in_var.get_shape()
        assert len(input_shape) == 4, 'Incoming Tensor shape must be 4-D'

        inference = tf.nn.avg_pool(in_var, kernel_size, strides, padding)

        return inference

def conv_2d(in_var, out_channels, filters=[3,3], strides=[1,1,1,1], 
            padding='SAME', name=None):
    assert name is not None, 'Op name should be specified'
    #
    with tf.name_scope(name):
        k_w, k_h = filters  # filter width/height
        W = tf.get_variable(name + "_W", [k_h, k_w, in_var.get_shape()[-1], out_channels],
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable(name + "_b", [out_channels], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(in_var, W, strides=strides, padding=padding)
        conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

        return conv

def linear(in_var, output_size, name=None, stddev=0.02, bias_val=0.0):
    assert name is not None, 'Op name should be specified'
    # 
    with tf.name_scope(name):
        shape = in_var.get_shape().as_list()
        W = tf.get_variable(name + "_W", [shape[1], output_size], tf.float32,
                              tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable(name + "_b", [output_size],
                             initializer=tf.constant_initializer(bias_val))
        return tf.matmul(in_var, W) + b

def residual_block(in_var, nb_blocks, out_channels, batch_norm=False, strides=[1,1,1,1],
                    downsample=False, downsample_strides=[1,2,2,1], name=None):
    assert name is not None, 'Op name should be specified'
    #
    with tf.name_scope(name):
        resnet = in_var
        in_channels = in_var.get_shape()[-1].value

        # multiple layers for a single residual block
        for i in xrange(nb_blocks):
            identity = resnet
            # apply convolution
            resnet = conv_2d(resnet, out_channels, 
                            strides=strides if not downsample else downsample_strides, 
                            name='{}_conv2d_{}'.format(name, i))
            # normalize batch before activations
            if batch_norm:
                resnet = batch_normalization(resnet)
            # apply activation function
            resnet = tf.nn.relu(resnet)
            # downsample
            if downsample:
                identity = avg_pool_2d(identity, strides=downsample_strides, name=name+'_avg_pool_2d')
            # projection to new dimension by padding
            if in_channels != out_channels:
                ch = (out_channels - in_channels)//2
                identity = tf.pad(identity, [[0, 0], [0, 0], [0, 0], [ch, ch]])
                in_channels = out_channels
            # add residual
            resnet = resnet + identity

        return resnet

def batch_normalization(in_var):
    with tf.name_scope('batch_normalization'):
        input_shape = in_var.get_shape()
        input_ndim = len(input_shape)

        axis = list(range(input_ndim - 1))
        mean, variance = tf.nn.moments(in_var, axis)
        inference = tf.nn.batch_normalization(in_var, mean, variance, 0.0, 1.0, 1e-5)
        inference.set_shape(input_shape)

        return inference


def residual_network(x):
    with tf.name_scope('residual_network'):
        _x = tf.reshape(x, [batch_size, 250, 250, 1])
        net = conv_2d(_x, 8, filters=[7,7], strides=[1,2,2,1], name='conv_0')
        net = max_pool(net, name='max_pool_0')
        net = residual_block(net, resnet_units, 8, name='resblock_1')
        net = residual_block(net, 1, 16, downsample=True, name='resblock_1-5')
        net = residual_block(net, resnet_units, 16, name='resblock_2')
        net = residual_block(net, 1, 32, downsample=True, name='resblock_2-5')
        net = residual_block(net, resnet_units, 32, name='resblock_3')
        # net = residual_block(net, 1, 64, downsample=True, name='resblock_3-5')
        # net = residual_block(net, resnet_units, 64, name='resblock_4')
        # batch normalize
        net = tf.nn.relu(net)
        net = global_avg_pool(net)
        return net

def compute_energy(o1, o2):
    with tf.name_scope('energy'):
        _energy = tf.reduce_sum(tf.pow(tf.sub(o1, o2), 2), reduction_indices=1)
        return _energy

def compute_loss(y_, o1, o2, energy, margin=None):
    with tf.name_scope('loss'):
        labels_t = y_
        labels_f = tf.sub(1.0, y_, name="1-y")
        M = tf.constant(margin, name="m")
        # compute loss_g and loss_i
        loss_g = tf.mul(0.5, tf.pow(energy, 2), name='l_G')
        loss_i = tf.mul(0.5, tf.pow(tf.maximum(0.0, tf.sub(M, energy)), 2), name='l_I')
        # compute full loss
        pos = tf.mul(labels_f, loss_g, name='1-Yl_G')
        neg = tf.mul(labels_t, loss_i, name='Yl_I')
        _loss = tf.reduce_mean(tf.add(pos, neg), name='loss')
        return _loss

def compute_accuracy(true_y, pred_y, margin):
    with tf.name_scope('accuracy'):
        _pred_y = tf.cast(tf.less(pred_y, margin/2), tf.float32)
        _acc = tf.reduce_mean(tf.mul(true_y, _pred_y))
        return _acc


# Create model
x1 = tf.placeholder(tf.float32, [None, 250, 250])
x2 = tf.placeholder(tf.float32, [None, 250, 250])

with tf.variable_scope("siamese") as scope:
    siamese1 = residual_network(x1)
    scope.reuse_variables()
    siamese2 = residual_network(x2)

# Calculate energy, loss, and accuracy
margin = 4.0
y_ = tf.placeholder(tf.float32, [None, 1])
energy_op = compute_energy(siamese1, siamese2)
loss_op = compute_loss(y_, siamese1, siamese2, energy_op, margin)
accuracy_op = compute_accuracy(y_, energy_op, margin)

# setup siamese network
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss_op)

# prepare data and tf.session
faces = ImageLoader()
sess = tf.Session()
summary_writer = tf.train.SummaryWriter("./tf_logs", graph=sess.graph)

sess.run(tf.initialize_all_variables())
for step in xrange(int(60e4)):
    batch_x1, batch_x2, batch_y1, batch_y2 = faces.next_batch(batch_size)
    batch_y = (batch_y1 == batch_y2).astype(np.float32)

    feed = {x1: batch_x1, x2: batch_x2, y_: batch_y}
    _, loss_v, acc_v = sess.run([train_step, loss_op, accuracy_op], feed_dict=feed)

    if np.isnan(loss_v):
        print('Model diverged with loss = NaN')
        quit()

    if step % 10 == 0:
        print ('step %d: loss %.3f accuracy: %.3f' % (step, loss_v, acc_v))
        sys.stdout.flush()


summary_writer.close()
sess.close()
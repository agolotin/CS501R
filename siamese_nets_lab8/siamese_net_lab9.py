from __future__ import print_function

from pdb import set_trace as debugger
from imageloader import ImageLoader
from tensorflow.python.training import moving_averages

import tensorflow as tf
import numpy as np

import os
import sys

batch_size = 6
resnet_units = 1

tf.reset_default_graph()

def global_avg_pool(in_var, name='global_pool'):
    assert name is not None, 'Op name should be specified'
    # start global average pooling
    with tf.name_scope(name):
        input_shape = in_var.get_shape()
        assert len(input_shape) == 4, 'Incoming Tensor shape must be 4-D'

        inference = tf.reduce_mean(in_var, [1, 2])
        return inference


def max_pool(in_var, kernel_size=[1,2,2,1], strides=[1,1,1,1], 
            padding='SAME', name=None):
    assert name is not None, 'Op name should be specified'
    assert strides is not None, 'Strides should be specified when performing max pooling'
    # start max pooling
    with tf.name_scope(name):
        input_shape = in_var.get_shape()
        assert len(input_shape) == 4, 'Incoming Tensor shape must be 4-D'

        inference = tf.nn.max_pool(in_var, kernel_size, strides, padding)
        return inference


def avg_pool_2d(in_var, kernel_size=[1,2,2,1], strides=None, 
                padding='SAME', name=None):
    assert name is not None, 'Op name should be specified'
    assert strides is not None, 'Strides should be specified when performing average pooling'
    # start average pooling
    with tf.name_scope(name):
        input_shape = in_var.get_shape()
        assert len(input_shape) == 4, 'Incoming Tensor shape must be 4-D'

        inference = tf.nn.avg_pool(in_var, kernel_size, strides, padding)
        return inference

def conv_2d(in_var, out_channels, filters=[3,3], strides=[1,1,1,1], 
            padding='SAME', name=None):
    assert name is not None, 'Op name should be specified'
    # start conv_2d
    with tf.name_scope(name):
        k_w, k_h = filters  # filter width/height
        W = tf.get_variable(name + "_W", [k_h, k_w, in_var.get_shape()[-1], out_channels],
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable(name + "_b", [out_channels], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(in_var, W, strides=strides, padding=padding)
        conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

        return conv

def residual_block(in_var, nb_blocks, out_channels, batch_norm=True, strides=[1,1,1,1],
                    downsample=False, downsample_strides=[1,2,2,1], name=None):
    assert name is not None, 'Op name should be specified'
    # start residual block
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
                resnet = batch_normalization(resnet, name=name+'batch_norm')
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

def batch_normalization(in_var, beta=0.0, gamma=1.0, epsilon=1e-5, decay=0.9, name=None):
    assert name is not None, 'Op name should be specified'
    # start batch normalization with moving averages
    input_shape = in_var.get_shape().as_list()
    input_ndim = len(input_shape)

    gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002)
    beta = tf.get_variable(name+'_beta', shape=[input_shape[-1]],
                           initializer=tf.constant_initializer(beta),
                           trainable=True)
    gamma = tf.get_variable(name+'_gamma', shape=[input_shape[-1]],
                            initializer=gamma_init, trainable=True)

    axis = list(range(input_ndim - 1))
    moving_mean = tf.get_variable(name+'_moving_mean',
                    input_shape[-1:], initializer=tf.zeros_initializer,
                    trainable=False)
    moving_variance = tf.get_variable(name+'_moving_variance',
                        input_shape[-1:], initializer=tf.ones_initializer,
                        trainable=False)

    # define a function to update mean and variance
    def update_mean_var():
        mean, variance = tf.nn.moments(in_var, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
        with tf.control_dependencies([update_moving_mean, update_moving_variance]):
            return tf.identity(mean), tf.identity(variance)

    # only update mean and variance with moving average while training
    mean, var = tf.cond(is_training, update_mean_var, lambda: (moving_mean, moving_variance))

    inference = tf.nn.batch_normalization(in_var, mean, var, beta, gamma, epsilon)
    inference.set_shape(input_shape)

    return inference


def residual_network(x):
    with tf.name_scope('residual_network') as scope:
        _x = tf.reshape(x, [batch_size, 250, 250, 1])
        net = conv_2d(_x, 8, filters=[7,7], strides=[1,2,2,1], name='conv_0')
        net = max_pool(net, name='max_pool_0')
        net = residual_block(net, resnet_units, 8, name='resblock_1')
        net = residual_block(net, 1, 16, downsample=True, name='resblock_1-5')
        net = residual_block(net, resnet_units, 16, name='resblock_2')
        net = residual_block(net, 1, 24, downsample=True, name='resblock_2-5')
        net = residual_block(net, resnet_units, 24, name='resblock_3')
        net = residual_block(net, 1, 32, downsample=True, name='resblock_3-5')
        net = residual_block(net, resnet_units, 32, name='resblock_4')
        net = batch_normalization(net, name='batch_norm')
        net = tf.nn.relu(net)
        net = global_avg_pool(net)
        return net

def compute_energy(o1, o2):
    with tf.name_scope('energy'):
        _energy = tf.reduce_sum(tf.abs(tf.sub(o1, o2)), reduction_indices=[1], keep_dims=True)
        return _energy

def compute_loss(y_, energy, margin):
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
        _pred_y = tf.cast(tf.less(pred_y, margin), tf.float32)
        _acc = tf.reduce_mean(tf.mul(true_y, _pred_y))
        return _acc


# Create model
x1 = tf.placeholder(tf.float32, [None, 250, 250])
x2 = tf.placeholder(tf.float32, [None, 250, 250])
is_training = tf.placeholder(tf.bool)

with tf.variable_scope("siamese") as scope:
    siamese1 = residual_network(x1)
    scope.reuse_variables()
    siamese2 = residual_network(x2)

# Calculate energy, loss, and accuracy
margin = 3.0
y_ = tf.placeholder(tf.float32, [None, 1])
energy_op = compute_energy(siamese1, siamese2)
loss_op = compute_loss(y_, energy_op, margin)
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

    feed = {x1: batch_x1, x2: batch_x2, y_: batch_y, is_training: True}
    _, loss_v, en, acc_v = sess.run([train_step, loss_op, energy_op, accuracy_op], feed_dict=feed)

    if np.isnan(loss_v):
        print('Model diverged with loss = NaN')
        sys.exit()
    if step % 10 == 0:
        print('step {0}: loss {1} accuracy: {2}'.format(step, loss_v, acc_v))
    if step % 1000 == 0:
        batch_x1, batch_x2, batch_y1, batch_y2 = faces.next_batch(1000, train=False)
        batch_y = (batch_y1 == batch_y2).astype(np.float32)

        feed = {x1: batch_x1, x2: batch_x2, y_: batch_y, is_training: False}
        t_acc = sess.run([accuracy_op], feed_dict=feed)
        print('train set accuracy: {0}'.format(t_acc))

    sys.stdout.flush()


summary_writer.close()
sess.close()
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from pdb import set_trace

mnist = input_data.read_data_sets( "MNIST_data/", one_hot=True )

# -------------------------------------------
# Global variables

batch_size = 128
z_dim = 10

# ==================================================================
# ==================================================================
# ==================================================================

def conv2d(in_var, output_dim, name="conv2d", reuse=None):
    k_w = 5  # filter width/height
    k_h = 5
    d_h = 2  # x,y strides
    d_w = 2
    ## make sure to reuse variables in discriminator scope
    with tf.variable_scope(name, reuse = reuse):
        W = tf.get_variable( "W", [k_h, k_w, in_var.get_shape()[-1], output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.02) )
        b = tf.get_variable( "b", [output_dim], initializer=tf.constant_initializer(0.0) )

        conv = tf.nn.conv2d( in_var, W, strides=[1, d_h, d_w, 1], padding='SAME' )
        conv = tf.reshape( tf.nn.bias_add( conv, b ), conv.get_shape() )

        return conv

def deconv2d(in_var, output_shape, name="deconv2d", stddev=0.02, bias_val=0.0):
    k_w = 5  # filter width/height
    k_h = 5
    d_h = 2  # x,y strides
    d_w = 2

    # [ height, width, in_channels, number of filters ]
    var_shape = [k_w, k_h, output_shape[-1], in_var.get_shape()[-1]]

    with tf.variable_scope(name):
        W = tf.get_variable("W", var_shape,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [output_shape[-1]],
                             initializer=tf.constant_initializer(bias_val))

        deconv = tf.nn.conv2d_transpose(in_var, W, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
    
        return deconv

def linear(in_var, output_size, name="linear", reuse=None, stddev=0.02, bias_val=0.0):
    shape = in_var.get_shape().as_list()

    ## make sure to reuse variables in discriminator scope
    with tf.variable_scope(name, reuse = reuse):
        W = tf.get_variable("W", [shape[1], output_size], tf.float32,
                              tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("b", [output_size],
                             initializer=tf.constant_initializer(bias_val))

        return tf.matmul(in_var, W) + b

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

# ==================================================================
# ==================================================================
# ==================================================================

# the generator should accept a (tensor of multiple) 'z' and return an image
# z will be [None,z_dim]

def generator_op(z):
	with tf.name_scope('g_h1'):
		# a linear layer, mapping z to 128*7*7 features
		g_h1_linear = linear(z, 128*7*7, name='g_h1_linear')
	# nonlinearity after the linear layer
	g_h1_relu = tf.nn.relu(g_h1_linear)
	# reshape to match deconvolution layer shape
	g_h1_reshaped = tf.reshape(g_h1_relu, shape=[batch_size, 7, 7, 128])
	with tf.name_scope('g_d2'):
		# a deconvolution layer, mapping H1 to a tensor that is [batch_size,14,14,128], followed by a relu
		g_d2_deconv = deconv2d(g_h1_reshaped, [batch_size, 14, 14, 128], name='g_d2_deconv')
	# nonlinearity after the deconvolution
	g_d2_relu = tf.nn.relu(g_d2_deconv)
	with tf.name_scope('g_d3'):
		# a deconvolution layer, mapping D2 to a tensor that is [batch_size,28,28,1]
		g_d3_deconv = deconv2d(g_d2_relu, [batch_size, 28, 28, 1], name='g_d3_deconv')
	# final output
	return tf.sigmoid(tf.reshape(g_d3_deconv, [batch_size, 784], name='g_d3_reshape'))

# -------------------------------------------
    
# the discriminator should accept a (tensor of muliple) images and
# return a probability that the image is real
# imgs will be [None,784]

def discriminator_op(imgs, reuse=None):
	# reshape input images to match the convolution layer
	d_reshaped_imgs = tf.reshape(imgs, [batch_size, 28, 28, 1], name='reshape_images')
	with tf.name_scope('d_h0'):
		# a 2d convolution on imgs with 32 filters, followed by a leaky relu
		d_h0_conv = lrelu(conv2d(d_reshaped_imgs, 32, name='d_h0_conv', reuse=reuse))
	with tf.name_scope('d_h1'):
		# a 2d convolution on H0 with 64 filters, followed by a leaky relu
		d_h1_conv = lrelu(conv2d(d_h0_conv, 64, name='d_h1_conv', reuse=reuse))
		# reshape for the linear layer
		d_h1_reshaped = tf.reshape(d_h1_conv, [batch_size, -1], 'd_h1_reshape')
	with tf.name_scope('d_h2'):
		# a linear layer from H1 to a 1024 dimensional vector
		d_h2_linear = linear(d_h1_reshaped, 1024, name='d_h2_linear', reuse=reuse)
	with tf.name_scope('d_h3'):
		# a linear layer mapping H2 to a single scalar (per image)
		d_h3_linear = linear(d_h2_linear, 1, name='d_h3_linear', reuse=reuse)
	# final output
	return tf.sigmoid(d_h3_linear)


# ==================================================================
# ==================================================================
# ==================================================================

with tf.name_scope('models'):
	with tf.name_scope('generator'):
		# randomly sampled vector that will be mapped to an image
		z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
		# pass the z variable into the generative model and get some fake images
		sample_images = generator_op(z)
	with tf.name_scope('discriminator'):
		true_images = tf.placeholder(tf.float32, shape=[None, 784], name='true_images')
		# pass some true images into the discriminator without reusing them first
		d_true_probs = discriminator_op(true_images, reuse=False)
		# pass some sampled images into the discriminator and reuse the same variables
		d_fake_probs = discriminator_op(sample_images, reuse=True)

with tf.name_scope('loss_functions'):
	with tf.name_scope('d_loss'):
		# discriminator loss function:
		#	maximize the log of the output probabilities on the true images 
		#	and the log of 1.0 - the output probabilities on the sampled images
		d_loss = tf.reduce_mean(-tf.add(tf.log(d_true_probs), tf.log(1 - d_fake_probs)))
	with tf.name_scope('g_loss'):
		# generator loss function:
		# maximize the log of the output probabilities on the sampled images
		g_loss = tf.reduce_mean(-tf.log(d_fake_probs))
	with tf.name_scope('d_acc'):
		# compute discriminator accuracy
		d_true_acc = tf.reduce_mean(tf.cast(tf.greater(d_true_probs, 0.5), tf.float32))
		d_fake_acc = tf.reduce_mean(tf.cast(tf.greater(d_fake_probs, 0.5), tf.float32))
		d_acc = tf.truediv(tf.add(d_true_acc, d_fake_acc), 2.0)

with tf.name_scope('optimizers'):
	with tf.name_scope('d_optimizer'):
		# Only optimize the discriminator variables for discriminator step
		d_vars = filter(lambda var: 'd_' in var.name, tf.trainable_variables())
		d_optim = tf.train.AdamOptimizer(2e-3, beta1=0.5).minimize(d_loss, var_list=d_vars)
	with tf.name_scope('g_optimizer'):
		# Only optimize the generator variables for generator step
		g_vars = filter(lambda var: 'g_' in var.name, tf.trainable_variables())
		g_optim = tf.train.AdamOptimizer(2e-3, beta1=0.5).minimize(g_loss, var_list=g_vars)

#
# ==================================================================
# ==================================================================
# ==================================================================
#

sess = tf.Session()
sess.run( tf.initialize_all_variables() )
summary_writer = tf.train.SummaryWriter( "./tf_logs", graph=sess.graph )

for i in range(500):
    batch = mnist.train.next_batch(batch_size)
    batch_images = batch[0]

    # train discriminator once
    sampled_zs = np.random.uniform(low=-1, high=1, size=(batch_size, z_dim)).astype(np.float32)
    sess.run(d_optim, feed_dict={ z:sampled_zs, true_images: batch_images })
    # train generator 3 times for every one generator step
    for _ in range(3):
        sampled_zs = np.random.uniform(low=-1, high=1, size=(batch_size, z_dim)).astype(np.float32)
        sess.run(g_optim, feed_dict={ z:sampled_zs })

    if i%10==0:
        d_acc_val, d_loss_val, g_loss_val = sess.run([d_acc, d_loss, g_loss],
                                                    feed_dict={ z:sampled_zs, true_images: batch_images })
        print "%d\t%.2f %.2f %.2f" % (i, d_loss_val, g_loss_val, d_acc_val)

summary_writer.close()

#  show some results
sampled_zs = np.random.uniform(-1, 1, size=(batch_size, z_dim)).astype(np.float32)
simgs = sess.run(sample_images, feed_dict={ z:sampled_zs })
simgs = simgs[0:64,:]

tiles = []
for i in range(0,8):
    tiles.append(np.reshape( simgs[i*8:(i+1)*8,:], [28*8,28]))
plt.imshow(np.hstack(tiles), interpolation='nearest', cmap=matplotlib.cm.gray)
plt.colorbar()
plt.show()
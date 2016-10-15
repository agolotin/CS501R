import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets( "MNIST_data/", one_hot=True )

# -------------------------------------------
# Global variables

batch_size = 128
z_dim = 10

# ==================================================================
# ==================================================================
# ==================================================================

def conv2d( in_var, output_dim, name="conv2d" ):
    k_w = 5  # filter width/height
    k_h = 5
    d_h = 2  # x,y strides
    d_w = 2

    with tf.variable_scope( name ):
        W = tf.get_variable( "W", [k_h, k_w, in_var.get_shape()[-1], output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.02) )
        b = tf.get_variable( "b", [output_dim], initializer=tf.constant_initializer(0.0) )

        conv = tf.nn.conv2d( in_var, W, strides=[1, d_h, d_w, 1], padding='SAME' )
        conv = tf.reshape( tf.nn.bias_add( conv, b ), conv.get_shape() )

        return conv

def deconv2d( in_var, output_shape, name="deconv2d", stddev=0.02, bias_val=0.0 ):
    k_w = 5  # filter width/height
    k_h = 5
    d_h = 2  # x,y strides
    d_w = 2

    # [ height, width, in_channels, number of filters ]
    var_shape = [ k_w, k_h, output_shape[-1], in_var.get_shape()[-1] ]

    with tf.variable_scope( name ):
        W = tf.get_variable( "W", var_shape,
                             initializer=tf.truncated_normal_initializer( stddev=0.02 ) )
        b = tf.get_variable( "b", [output_shape[-1]],
                             initializer=tf.constant_initializer( bias_val ))

        deconv = tf.nn.conv2d_transpose( in_var, W, output_shape=output_shape, strides=[1, d_h, d_w, 1] )
        deconv = tf.reshape( tf.nn.bias_add( deconv, b), deconv.get_shape() )
    
        return deconv

def linear( in_var, output_size, name="linear", stddev=0.02, bias_val=0.0 ):
    shape = in_var.get_shape().as_list()

    with tf.variable_scope( name ):
        W = tf.get_variable( "W", [shape[1], output_size], tf.float32,
                              tf.random_normal_initializer( stddev=stddev ) )
        b = tf.get_variable( "b", [output_size],
                             initializer=tf.constant_initializer( bias_val ))

        return tf.matmul( in_var, W ) + b

def lrelu( x, leak=0.2, name="lrelu" ):
    return tf.maximum( x, leak*x )

# ==================================================================
# ==================================================================
# ==================================================================

# the generator should accept a (tensor of multiple) 'z' and return an image
# z will be [None,z_dim]

def gen_model(z):
	with tf.variable_scope('generator'):
		# a linear layer, mapping z to 128*7*7 features, followed by a relu
		g_h1_linear = tf.nn.relu(linear(z, 128*7*7, name='g_h1_linear'))
		# reshape to match deconvolution layer shape
		g_h1_reshaped = tf.reshape(g_h1_linear, shape=[batch_size, 7, 7, 128])
		# a deconvolution layer, mapping H1 to a tensor that is [batch_size,14,14,128], followed by a relu
		g_d2_deconv = tf.nn.relu(deconv2d(g_h1_reshaped, [batch_size, 14, 14, 128], name='g_d2_deconv'))
		# a deconvolution layer, mapping D2 to a tensor that is [batch_size,28,28,1]
		g_d3_deconv = deconv2d(g_d2_deconv, [batch_size, 28, 28, 1], name='g_d3_deconv')
		# final output
		return tf.sigmoid(tf.reshape(g_d3_deconv, [batch_size, 784]))

# -------------------------------------------
    
# the discriminator should accept a (tensor of muliple) images and
# return a probability that the image is real
# imgs will be [None,784]

def disc_model(imgs):
	with tf.variable_scope('discriminator'):
		# reshape input images to match the convolution layer
		d_reshaped_imgs = tf.reshape(imgs, [batch_size, 28, 28, 1])
		# a 2d convolution on imgs with 32 filters, followed by a leaky relu
		d_h0_conv = lrelu(conv2d(d_reshaped_imgs, 32, name='d_h0_conv'))
		# a 2d convolution on H0 with 64 filters, followed by a leaky relu
		d_h1_conv = lrelu(conv2d(d_h0_conv, 64, name='d_h1_conv'))
		# a linear layer from H1 to a 1024 dimensional vector
		d_reshaped_h1 = tf.reshape(d_h1_conv, [batch_size, -1])
		d_h2_linear = linear(d_reshaped_h1, 1024, name='d_h2_linear')
		# a linear layer mapping H2 to a single scalar (per image)
		d_h3_linear = linear(d_h2_linear, 1, name='d_h3_linear')
		# final output
		return tf.sigmoid(d_h3_linear)


# ==================================================================
# ==================================================================
# ==================================================================

# Placeholders should be named 'z' and ''true_images'
# Training ops should be named 'd_optim' and 'g_optim'
# The output of the generator should be named 'sample_images'
z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
true_images = tf.placeholder(tf.float32, shape=[None, 784], name='true_images')

# -------------------------------------------
# pass the z variable into the generative model
sample_images = gen_model(z)
with tf.variable_scope('discriminator') as disc_scope:
	# pass some true images into the discriminator
	d_true_probs = disc_model(true_images)
	# make sure to reuse variables
	disc_scope.reuse_variables()
	# pass some sampled images into the discriminator
	d_fake_probs = disc_model(sample_images)
# discriminator loss function:
#	maximize the log of the output probabilities on the true images 
#	and the log of 1.0 - the output probabilities on the sampled images
d_loss = -tf.reduce_sum(tf.add(tf.log(d_true_probs), tf.sub(tf.log(d_fake_probs), tf.constant(1, dtype=tf.float32))))
# generator loss function:
# maximize the log of the output probabilities on the sampled images
g_loss = -tf.reduce_sum(tf.log(d_fake_probs))
# compute discriminator accuracy
true_acc = tf.reduce_mean(tf.cast(tf.greater(d_true_probs, 0.5), tf.float32))
fake_acc = tf.reduce_mean(tf.cast(tf.greater(d_fake_probs, 0.5), tf.float32))
d_acc = (true_acc + fake_acc) / 2.0

# -------------------------------------------
# Only optimize the discriminator variables for discriminator step
d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
d_optim = tf.train.AdamOptimizer(2e-3, beta1=0.5).minimize(d_loss, var_list=d_vars)
# Only optimize the generator variables for generator step
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
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
        sess.run( g_optim, feed_dict={ z:sampled_zs } )

    if i%10==0:
        d_acc_val,d_loss_val,g_loss_val = sess.run([d_acc,d_loss,g_loss],
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
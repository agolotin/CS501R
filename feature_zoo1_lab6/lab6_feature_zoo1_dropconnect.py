
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from tensorflow.examples.tutorials.mnist import input_data

#
# ==================================================================
#

def weight_variable(shape):
  initial = tf.truncated_normal( shape, stddev=0.1 )
  return tf.Variable( initial )

def bias_variable(shape):
  initial = tf.constant( 0.1, shape=shape )
  return tf.Variable(initial)

#
# ==================================================================
#

# Declare computation graph

y_ = tf.placeholder( tf.float32, shape=[None, 10], name="y_" )
x = tf.placeholder( tf.float32, [None, 784], name="x" )
keep_probability = tf.placeholder(tf.float32)
scale = tf.placeholder(tf.float32)

####### H1 #########
W1 = weight_variable([784, 500])
b1 = bias_variable([500])
# Dropconnect
w_M1 = tf.to_float(tf.less(tf.random_uniform(tf.shape(W1)), keep_probability))
b_M1 = tf.to_float(tf.less(tf.random_uniform(tf.shape(b1)), keep_probability))
W1_prime = tf.mul(W1, w_M1)
b1_prime = tf.mul(b1, b_M1)
# Nonlinearity
h1 = tf.nn.relu(tf.matmul( x, W1_prime ) + b1_prime)

####### H2 #########
W2 = weight_variable([500, 500])
b2 = bias_variable([500])
# Dropconnect
w_M2 = tf.to_float(tf.less(tf.random_uniform(tf.shape(W2)), keep_probability))
b_M2 = tf.to_float(tf.less(tf.random_uniform(tf.shape(b2)), keep_probability))
W2_prime = tf.mul(W2, w_M2)
b2_prime = tf.mul(b2, b_M2)
# Nonlinearity
h2 = tf.nn.relu(tf.matmul( h1, W2_prime ) + b2_prime)

####### H3 #########
W3 = weight_variable([500, 1000])
b3 = bias_variable([1000])
# Dropconnect
w_M3 = tf.to_float(tf.less(tf.random_uniform(tf.shape(W3)), keep_probability))
b_M3 = tf.to_float(tf.less(tf.random_uniform(tf.shape(b3)), keep_probability))
W3_prime = tf.mul(W3, w_M3)
b3_prime = tf.mul(b3, b_M3)
# Nonlinearity
h3 = tf.nn.relu(tf.matmul( h2, W3_prime ) + b3_prime)

W4 = weight_variable([1000, 10])
b4 = bias_variable([10])
# Scale the activations by scale variable 
y_hat = tf.mul(tf.nn.softmax(tf.matmul(h3, W4) + b4), scale)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_hat), reduction_indices=[1]))
xent_summary = tf.scalar_summary( 'xent', cross_entropy )

correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc_summary = tf.scalar_summary( 'accuracy', accuracy )

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#
# ==================================================================
#

# NOTE: we're using a single, fixed batch of the first 1000 images
mnist = input_data.read_data_sets( "MNIST_data/", one_hot=True )

images = mnist.train.images[ 0:1000, : ]
labels = mnist.train.labels[ 0:1000, : ]

keep_probs = [ 0.1, 0.25, 0.5, 0.75, 1.0 ]

train_acc = []
test_acc = []
sess = tf.Session()
for p in keep_probs:
    sess.run( tf.initialize_all_variables() )
    for i in tqdm(range(1500)):
        sess.run( train_step, feed_dict={ x: images, y_: labels, keep_probability: p, scale: p} )
        # print( "step %d, training accuracy %g" % (i, acc) )
    t_acc = sess.run(accuracy, feed_dict={ x: images, y_: labels, keep_probability: 1.0, scale: p })
    train_acc.append(t_acc)

#
# ==================================================================
#

    final_acc = sess.run( accuracy, feed_dict={ x: mnist.test.images, y_: mnist.test.labels, keep_probability: 1.0, scale: p } )
    test_acc.append(final_acc)
    print( "train accuracy %g" % t_acc )
    print( "test accuracy %g" % final_acc )


import matplotlib.pyplot as plt

plt.plot(keep_probs, train_acc, label='Training Accuracy')
plt.plot(keep_probs, test_acc, label='Testing Accuracy')
plt.axhline(y=0.8465, color='r', ls='dashed', label='Baseline')
plt.title('Dropconnect Plot')
plt.ylabel("Classification Accuracy (%)")
plt.xlabel("Keep Probability")
plt.legend(loc='best')
plt.show()


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
lambda_ = tf.placeholder(tf.float32)

W1 = weight_variable([784, 500])
b1 = bias_variable([500])
h1_reg = tf.reduce_prod(tf.abs(W1)) + tf.reduce_prod(tf.abs(b1))
h1 = tf.nn.relu(tf.matmul( x, W1 ) + b1)

W2 = weight_variable([500, 500])
b2 = bias_variable([500])
h2_reg = h1_reg + tf.reduce_prod(tf.abs(W2)) + tf.reduce_prod(tf.abs(b2))
h2 = tf.nn.relu(tf.matmul( h1, W2 ) + b2)

W3 = weight_variable([500, 1000])
b3 = bias_variable([1000])
h3_reg = h2_reg + tf.reduce_prod(tf.abs(W3)) + tf.reduce_prod(tf.abs(b3))
h3 = tf.nn.relu(tf.matmul( h2, W3 ) + b3)

W4 = weight_variable([1000, 10])
b4 = bias_variable([10])
final_reg = h3_reg + tf.reduce_prod(tf.abs(W4)) + tf.reduce_prod(tf.abs(b4))
y_hat = tf.nn.softmax(tf.matmul(h3, W4) + b4)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_hat), reduction_indices=[1]))
regularized_ce = cross_entropy + lambda_ * final_reg
xent_summary = tf.scalar_summary( 'xent', cross_entropy )
reg_ce_summary = tf.scalar_summary( 'reg_ce', regularized_ce )

correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc_summary = tf.scalar_summary( 'accuracy', accuracy )

train_step = tf.train.AdamOptimizer(1e-4).minimize(regularized_ce)

# NOTE: we're using a single, fixed batch of the first 1000 images
mnist = input_data.read_data_sets( "MNIST_data/", one_hot=True )

images = mnist.train.images[ 0:1000, : ]
labels = mnist.train.labels[ 0:1000, : ]

_lambda = [0.1, 0.01, 0.001]

train_acc = []
test_acc = []
sess = tf.Session()
for l in _lambda:
    sess.run(tf.initialize_all_variables())
    for _ in tqdm(range(150)):
        sess.run(train_step, feed_dict={ x: images, y_: labels, lambda_: l } )
        #print( "step %d, training accuracy %g" % (i, acc) )

    t_acc = sess.run(accuracy, feed_dict={ x: images, y_: labels, lambda_: l })
    final_acc = sess.run( accuracy, feed_dict={ x: mnist.test.images, y_: mnist.test.labels, lambda_: l } )
    train_acc.append(t_acc)
    test_acc.append(final_acc)
    print( "train accuracy %g" % t_acc )
    print( "test accuracy %g" % final_acc )

import matplotlib.pyplot as plt

plt.plot(_lambda, train_acc, label='Training Accuracy')
plt.plot(_lambda, test_acc, label='Testing Accuracy')
plt.axhline(y=0.8465, color='r', ls='dashed', label='Baseline')
plt.title('Dropout Plot')
plt.ylabel("Classification Accuracy (%)")
plt.xlabel("Keep Probability")
plt.legend(loc='best')
plt.show()

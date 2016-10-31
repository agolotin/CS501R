#! /usr/bin/env python

from pdb import set_trace as debugger

from textloader import TextLoader
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops.rnn_cell import BasicLSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell

import tensorflow as tf
import numpy as np


'''
class GRUCell(RNNCell):
 
    def __init__(self, num_units):
        self._num_units
 
    @property
    def state_size(self):
        self._num_units
 
    @property
    def output_size(self):
        self._num_units
 
    def __call__(self, inputs, state, scope=None):
        pass
'''

#
# -------------------------------------------
#
# Global variables

batch_size = 50
sequence_length = 50

data_loader = TextLoader(".", batch_size, sequence_length)

vocab_size = data_loader.vocab_size  # dimension of one-hot encodings
state_dim = 128

num_layers = 2

#
# ==================================================================
# ==================================================================
# ==================================================================
#

# define placeholders for our inputs.  
# in_ph is assumed to be [batch_size,sequence_length]
# targ_ph is assumed to be [batch_size,sequence_length]

in_ph = tf.placeholder(tf.int32, [batch_size, sequence_length], name='inputs')
targ_ph = tf.placeholder(tf.int32, [batch_size, sequence_length], name='targets')
in_onehot = tf.one_hot(in_ph, vocab_size, name="input_onehot")

inputs = tf.split(1, sequence_length, in_onehot)
inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
targets = tf.split(1, sequence_length, targ_ph)

# at this point, inputs is a list of length sequence_length
# each element of inputs is [batch_size,vocab_size]

# targets is a list of length sequence_length
# each element of targets is a 1D vector of length batch_size

# ------------------
# COMPUTATION GRAPH

with tf.variable_scope('lstm', reuse=None):
    # create a BasicLSTMCell
    #   use it to create a MultiRNNCell
    #   use it to create an initial_state
    #     initial_state will be a *list* of tensors
    lstm_cell = BasicLSTMCell(state_dim, state_is_tuple=True)
    rnn_cells = MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)
    initial_state = rnn_cells.zero_state(batch_size, tf.float32)

    # call seq2seq.rnn_decoder
    outputs, final_state = seq2seq.rnn_decoder(inputs, initial_state, rnn_cells)

    # transform the list of state outputs to a list of logits.
    # use a linear transformation.
    trainable_weights = tf.get_variable('trainable_w', shape=[state_dim, vocab_size],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    logits = [tf.matmul(output, trainable_weights) for output in outputs]

    # call seq2seq.sequence_loss
    weights = [tf.ones(shape=[batch_size], dtype=tf.float32)] * len(logits)
    loss = seq2seq.sequence_loss(logits, targets, weights)

    # create a training op using the Adam optimizer
    optim = tf.train.AdamOptimizer(0.01).minimize(loss)


# ------------------
# SAMPLER GRAPH

# place your sampler graph here it will look a lot like your
# computation graph, except with a "batch_size" of 1.

# remember, we want to reuse the parameters of the cell and whatever
# parameters you used to transform state outputs to logits!

s_inputs = tf.placeholder(tf.int32, [1], name='x_')
s_input = [tf.one_hot(s_inputs, vocab_size, name="x_onehot")]

#_input = tf.split(1, 1, s_onehot)
#s_input = [tf.squeeze(_input, [1])]

with tf.variable_scope('lstm', reuse=True):
    s_lstm_cell = BasicLSTMCell(state_dim, state_is_tuple=True)
    s_rnn_cells = MultiRNNCell([s_lstm_cell] * num_layers, state_is_tuple=True)
    s_initial_state = rnn_cells.zero_state(1, tf.float32)

    # call seq2seq.rnn_decoder
    s_outputs, s_final_state = seq2seq.rnn_decoder(s_input, s_initial_state, s_rnn_cells)

    # transform the list of state outputs to a list of logits.
    # use a linear transformation.
    s_weights = tf.get_variable('trainable_w', shape=[state_dim, vocab_size],
        initializer=tf.random_normal_initializer(stddev=0.1))
    s_probs = tf.nn.softmax(tf.matmul(s_outputs[0], s_weights))

#
# ==================================================================
# ==================================================================
# ==================================================================
#

def sample(num=200, prime='ab'):

    # prime the pump 

    # generate an initial state. this will be a list of states, one for
    # each layer in the multicell.
    s_state = sess.run(s_initial_state)

    # for each character, feed it into the sampler graph and
    # update the state.
    for char in prime[:-1]:
        x = np.ravel(data_loader.vocab[char]).astype('int32')
        feed = {s_inputs:x}
        for i, s in enumerate(s_initial_state):
            feed[s] = s_state[i]
        s_state = sess.run(s_final_state, feed_dict=feed)

    # now we have a primed state vector; we need to start sampling.
    ret = prime
    char = prime[-1]
    for n in range(num):
        x = np.ravel(data_loader.vocab[char]).astype('int32')

        # plug the most recent character in...
        feed = {s_inputs:x}
        for i, s in enumerate(s_initial_state):
            feed[s] = s_state[i]
        ops = [s_probs]
        ops.extend(list(s_final_state))

        retval = sess.run(ops, feed_dict=feed)

        s_probsv = retval[0]
        s_state = retval[1:]

        # ...and get a vector of probabilities out!

        # now sample (or pick the argmax)
        # sample = np.argmax( s_probsv[0] )
        sample = np.random.choice(vocab_size, p=s_probsv[0])

        pred = data_loader.chars[sample]
        ret += pred
        char = pred

    return ret

#
# ==================================================================
# ==================================================================
# ==================================================================
#

sess = tf.Session()
sess.run(tf.initialize_all_variables())
summary_writer = tf.train.SummaryWriter("./tf_logs", graph=sess.graph)

lts = []

print "FOUND %d BATCHES" % data_loader.num_batches

for j in range(1000):

    state = sess.run(initial_state)
    data_loader.reset_batch_pointer()

    for i in range(data_loader.num_batches):
        
        x,y = data_loader.next_batch()
        # we have to feed in the individual states of the MultiRNN cell
        feed = {in_ph: x, targ_ph: y}
        for k, st in enumerate(initial_state):
            feed[st] = state[k]
        ops = [optim, loss]
        ops.extend(list(final_state))

        # retval will have at least 3 entries:
        # 0 is None (triggered by the optim op)
        # 1 is the loss
        # 2+ are the new final states of the MultiRNN cell
        _retval = sess.run(ops, feed_dict=feed)

        lt = _retval[1]
        state = _retval[2:]

        if i%10 == 0:
            print "%d %d\t%.4f" % (j, i, lt)
            lts.append(lt)

    print sample(num=60, prime="And ")
#    print sample( num=60, prime="ababab" )
#    print sample( num=60, prime="foo ba" )
#    print sample( num=60, prime="abcdab" )

summary_writer.close()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

#import matplotlib
#import matplotlib.pyplot as plt
#plt.plot( lts )
#plt.show()

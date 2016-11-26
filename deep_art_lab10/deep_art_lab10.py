from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import vgg16

from scipy.misc import imread, imresize, imsave
from pdb import set_trace as debugger

content_wegiht = tf.constant(1e-4, dtype=tf.float32)
style_weight = tf.constant(1e-1, dtype=tf.float32)
content_layers = ['conv4_2']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

sess = tf.Session()
summary_writer = tf.train.SummaryWriter("./tf_logs", graph=sess.graph)

opt_img = tf.Variable(tf.truncated_normal([1,224,224,3], 
                                           dtype=tf.float32, 
                                           stddev=1e-1), 
                                           name='opt_img')

tmp_img = tf.clip_by_value(opt_img, 0.0, 255.0)

vgg = vgg16.vgg16(tmp_img, 'vgg16_weights.npz', sess)

style_img = imread('style.png', mode='RGB')
style_img = imresize(style_img, (224, 224))
style_img = np.reshape(style_img, [1,224,224,3])

content_img = imread('content.png', mode='RGB')
content_img = imresize(content_img, (224, 224))
content_img = np.reshape(content_img, [1,224,224,3])

layers = [ 'conv1_1', 'conv1_2',
           'conv2_1', 'conv2_2',
           'conv3_1', 'conv3_2', 'conv3_3',
           'conv4_1', 'conv4_2', 'conv4_3',
           'conv5_1', 'conv5_2', 'conv5_3' ]

ops = [getattr(vgg, x) for x in layers]

content_acts = sess.run(ops, feed_dict={vgg.imgs: content_img})
style_acts = sess.run(ops, feed_dict={vgg.imgs: style_img})

#
# --- construct cost function 
#


def total_loss_op(content_l, style_l):
    return tf.add(tf.mul(content_wegiht, content_l), tf.mul(style_weight, style_l))


with tf.name_scope('content'):
    # get content features op from vgg activations
    content_ops = [getattr(vgg, x) for x in content_layers]
    # loss op
    content_loss = tf.nn.l2_loss(tf.sub(content_acts[8], content_ops))

with tf.name_scope('style'):
    main_style_acts = map(lambda idx: style_acts[idx], [0, 2, 4, 7, 10])
    # get style features op from vgg activations
    style_ops = [getattr(vgg, x) for x in style_layers]
    w_l = tf.constant(1/5, dtype=tf.float32, name='factor')
    style_loss = []
    for i in xrange(len(style_ops)):
        # get dimentions
        N = style_ops[i].get_shape()[1].value
        M = style_ops[i].get_shape()[1].value * style_ops[i].get_shape()[2].value
        # compute gram matrices for style layers
        _layer_acts = main_style_acts[i].reshape(-1, N**2)
        main_style_grams = np.dot(_layer_acts.T, _layer_acts)
        # compute gram matrix for generated image for a layer
        _gram_matrix = tf.reshape(style_ops[i], [-1, N**2])
        g_l = tf.matmul(tf.transpose(_gram_matrix), _gram_matrix)
        # compute style loss
        e_l = tf.nn.l2_loss(tf.sub(g_l, main_style_grams))
        e_l = tf.truediv(e_l, 2.0*(N**2)*(M**2), name='e_l') # 4N**2M**2
        style_loss.append(tf.mul(w_l, e_l, name='we_l'))
    # sum all the style losses together
    style_loss = reduce(tf.add, style_loss)
    # compute gram matrices for generated image
    #_gram_matrices = map(lambda layer: tf.reshape(layer, [-1, layer.get_shape()[-1].value]), style_ops)
    #g_ls = map(lambda _gram: tf.matmul(tf.transpose(_gram), _gram), _gram_matrices)
    # compute style losses
    #e_ls = map(lambda idx: tf.nn.l2_loss(tf.sub(g_ls[idx], main_style_grams[idx]), name='e_L'), range(len(g_ls)))
    #intermediate_style_losses = map(lambda e_l: tf.mul(w_l, e_l, name='we_l'), e_ls)
    #style_loss = tf.truediv(reduce(tf.add, intermediate_style_losses), 2.0) # 4N**2M**2

with tf.name_scope('loss'):
    total_loss = total_loss_op(content_loss, style_loss)

# Relevant snippets from the paper:
#   For the images shown in Fig 2 we matched the content representation on layer 'conv4_2'
#   and the style representations on layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1' and 'conv5_1'
#   The ratio alpha/beta was  1x10-3
#   The factor w_l was always equal to one divided by the number of active layers (ie, 1/5)

# --- place your adam optimizer call here
#     (don't forget to optimize only the opt_img variable)
train_step = tf.train.AdamOptimizer(1e-1).minimize(total_loss)

# this clobbers all VGG variables, but we need it to initialize the
# adam stuff, so we reload all of the weights...
sess.run(tf.initialize_all_variables())
vgg.load_weights('vgg16_weights.npz', sess)

# initialize with the content image
sess.run(opt_img.assign(content_img))

def _imsave(path, img):
    img = img.astype(np.uint8)
    imsave(path, img)

# --- optimization loop
for step in xrange(int(10e2)):
    # take an optimizer step
    _, _loss = sess.run([train_step, total_loss])
    # clip values
    if step % 100 == 0:
        _img = sess.run(opt_img)
        _img = np.clip(_img, 0.0, 255.0)
        _imsave('output/img_{}'.format(step), _img)
        sess.run(opt_img.assign(_img))

    print('iteration {} loss {}'.format(step, _loss))


sess.close()
summary_writer.close()
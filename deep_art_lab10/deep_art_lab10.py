from __future__ import print_function
from __future__ import division

assert len(sys.argv) == 2, 'Specify output image name'

import numpy as np
import tensorflow as tf
import vgg16
import sys

from scipy.misc import imread, imresize, imsave
from pdb import set_trace as debugger

content_wegiht = tf.constant(1, dtype=tf.float32)
style_weight = tf.constant(10000, dtype=tf.float32)

content_layer = 'conv4_2'
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

sess = tf.Session()
summary_writer = tf.train.SummaryWriter("./tf_logs", graph=sess.graph)

opt_img = tf.Variable(tf.truncated_normal([1,224,224,3], dtype=tf.float32, 
                                           stddev=1e-1), name='opt_img')
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
with tf.name_scope('content'):
    assert isinstance(content_layer, str)
    # get content features op from vgg activations
    p_content_op = content_acts[layers.index(content_layer)]
    g_content_op = getattr(vgg, content_layer)
    # loss op
    content_loss = tf.nn.l2_loss(tf.sub(g_content_op, p_content_op))

with tf.name_scope('precompute_style'):
    style_grams = []
    style_layer_indices = map(lambda _l: layers.index(_l), style_layers)
    for i in style_layer_indices:
        curr_style_acts = style_acts[i]
        # compute gram matrices for style layers
        depth = curr_style_acts.shape[-1]
        _layer_acts = curr_style_acts.reshape(-1, depth)
        _gram = np.dot(_layer_acts.T, _layer_acts)
        style_grams.append(_gram)

with tf.name_scope('style'):
    # get style features op from vgg activations
    style_ops = [getattr(vgg, x) for x in style_layers]
    w_l = tf.constant(1/5.0, dtype=tf.float32, name='factor')
    style_losses = []
    for i in xrange(len(style_ops)):
        # get dimentions
        _, width, height, depth = map(lambda x: x.value, style_ops[i].get_shape())
        N = depth
        M = width * height
        # compute gram matrix for generated image for a layer
        _reshaped_layer = tf.reshape(style_ops[i], [-1, depth])
        g_l = tf.matmul(_reshaped_layer, _reshaped_layer, transpose_a=True)
        # compute style loss
        e_l = tf.nn.l2_loss(tf.sub(g_l, style_grams[i]))
        e_l = tf.truediv(e_l, 2.0*(N**2)*(M**2), name='e_l')
        style_losses.append(tf.mul(w_l, e_l, name='we_l'))
    # sum all the style losses together
    style_loss = reduce(tf.add, style_losses)

with tf.name_scope('loss'):
    total_loss = tf.add(content_wegiht * content_loss, style_weight * style_loss)

# Relevant snippets from the paper:
#   For the images shown in Fig 2 we matched the content representation on layer 'conv4_2'
#   and the style representations on layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1' and 'conv5_1'
#   The ratio alpha/beta was  1x10-3
#   The factor w_l was always equal to one divided by the number of active layers (ie, 1/5)

train_step = tf.contrib.opt.ScipyOptimizerInterface(total_loss, var_list=[opt_img], method='L-BFGS-B', options={'maxiter': 0})

# this clobbers all VGG variables, but we need it to initialize the
# adam stuff, so we reload all of the weights...
sess.run(tf.initialize_all_variables())
vgg.load_weights('vgg16_weights.npz', sess)

# initialize with the content image
sess.run(opt_img.assign(content_img))

def _imsave(path, img):
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    #img = imresize(img, (1014, 1280, 3)).astype(np.uint8)
    imsave(path, img)

# optimization loop
for step in xrange(int(10e4)):
    # report loss
    if step % 10 == 0:
        _loss, _cont, _style = sess.run([total_loss, content_loss, style_loss])
        print('iteration {} total loss {}; style loss {}; content loss {}'.format(step, _loss, _style, _cont))
    # take an optimizer step
    train_step.minimize(sess)
    # clip values
    if step % 100 == 0:
        _img = sess.run(opt_img)
        _imsave('output/{}_{}.png'.format(sys.argv[1], step), _img[0])
        _img = tf.clip_by_value(_img, 0.0, 255.0)
        sess.run(opt_img.assign(_img))


sess.close()
summary_writer.close()
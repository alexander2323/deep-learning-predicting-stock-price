
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy
def prelu(_x):
    alphas = tf.Variable(0.1*tf.ones(_x.get_shape()[-1]))
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg

def Inception1D(inputs, filters):
    conv1 = tf.layers.conv1d(inputs=inputs, filters=filters, kernel_size=1, padding = 'same')
    conv1 = prelu(conv1)
    conv1 = tf.contrib.layers.batch_norm(inputs = conv1)
    
    preconv3 = tf.layers.conv1d(inputs=inputs, filters=filters, kernel_size=1, padding = 'same')
    preconv3 = prelu(preconv3)
    preconv3 = tf.contrib.layers.batch_norm(inputs = preconv3)
    conv3 = tf.layers.conv1d(inputs=preconv3, filters=filters, kernel_size=3, padding = 'same')
    conv3 = prelu(conv3)
    conv3 = tf.contrib.layers.batch_norm(inputs = conv3)
    
    preconv5 = tf.layers.conv1d(inputs=inputs, filters=filters, kernel_size=1, padding = 'same')
    preconv5 = prelu(preconv5)
    preconv5 = tf.contrib.layers.batch_norm(inputs = preconv5)
    conv5 = tf.layers.conv1d(inputs=preconv5, filters=filters, kernel_size=5, padding = 'same')
    conv5 = prelu(conv5)
    conv5 = tf.contrib.layers.batch_norm(inputs = conv5)
    
    out = tf.concat([conv1, conv3, conv5], axis=3)
    return out

def Xception1D(inputs, filters):
    conv1 = tf.layers.separable_conv1d(inputs=inputs, filters=filters, kernel_size=1, padding = 'same')
    conv1 = prelu(conv1)
    conv1 = tf.contrib.layers.batch_norm(inputs = conv1)
    
    preconv3 = tf.layers.conv1d(inputs=inputs, filters=filters, kernel_size=1, padding = 'same')
    preconv3 = prelu(preconv3)
    preconv3 = tf.contrib.layers.batch_norm(inputs = preconv3)
    conv3 = tf.layers.separable_conv1d(inputs=preconv3, filters=filters, kernel_size=3, padding = 'same')
    conv3 = prelu(conv3)
    conv3 = tf.contrib.layers.batch_norm(inputs = conv3)
    
    preconv5 = tf.layers.conv1d(inputs=inputs, filters=filters, kernel_size=1, padding = 'same')
    preconv5 = prelu(preconv5)
    preconv5 = tf.contrib.layers.batch_norm(inputs = preconv5)
    conv5 = tf.layers.separable_conv1d(inputs=preconv5, filters=filters, kernel_size=5, padding = 'same')
    conv5 = prelu(conv5)
    conv5 = tf.contrib.layers.batch_norm(inputs = conv5)
    
    out = tf.concat([conv1, conv3, conv5], axis=3)
    return out

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = numpy.empty(a.shape, dtype=a.dtype)
    shuffled_b = numpy.empty(b.shape, dtype=b.dtype)
    permutation = numpy.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def L2_loss(weights, beta=1e-4):
    norm = tf.reduce_sum([ tf.nn.l2_loss(v) for v in weights ], axis=0)
    return norm*beta

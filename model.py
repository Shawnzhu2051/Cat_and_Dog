from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

import tensorflow.python.platform
import six
import tensorflow as tf
import handle_inputs

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/Users/ComingWind/Documents/2017/ML/cat_and_dog_processed_1/',
                           """Path to the record data directory.""")

image_train_size = handle_inputs.IMAGE_AMOUNTS_FOR_TRAINING
image_eval_size = handle_inputs.IMAGE_AMOUNTS_FOR_EVAL
image_size = 32 * 32

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

def _variables_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variables_with_decay(name, shape, stddev, wd):
    var = _variables_on_cpu(name, shape, tf.truncated_normal_initializer(stddev))

    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_loss")
        tf.add_to_collection('losses', weight_decay)
    return var

def distord_inputs():
    data_dir = os.path.join(FLAGS.data_dir)
    return handle_inputs.distorted_inputs(data_dir=data_dir, batch_size=FLAGS.batch_size)

def inputs(eval_data):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data direction')
    data_dir = os.path.join(FLAGS.data_dir)
    return handle_inputs.inputs(eval_data=eval_data, data_dir=data_dir, batch_size=FLAGS.batch_size)

def model_build(images):
    #first convolution layer conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variables_with_decay('weights', shape=[5, 5, 3, 64], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding='SAME', use_cudnn_on_gpu=False)
        biases = _variables_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        #print(conv1)

    #first pooling layer pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    #print(pool1)

    #second convlution layer conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variables_with_decay('weights', shape=[5,5,64,64], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(pool1, kernel, [1,1,1,1], padding='SAME')
        biases = _variables_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        #print(conv2)

    #second pooling layer pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
    #print(pool2)

    #third convolution layer conv3
    with tf.variable_scope('conv3') as scope:
        kernel = _variables_with_decay('weights', shape=[5,5,64,64], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(pool2, kernel, [1,1,1,1], padding='SAME')
        biases = _variables_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        #print(conv3)

    #third pooling layer pool3
    pool3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool3')
    #print(pool3)

    #first full connected layer fc1
    weights = _variables_on_cpu('fc_weight1', [4 * 4 * 64, 1024], tf.constant_initializer(0.0))
    bias = _variables_on_cpu('biases_fc1', [1024], tf.constant_initializer(0.0))
    image_fc1_flat = tf.reshape(pool3, [-1, 4*4*64])
    fc1 = tf.nn.relu(tf.matmul(image_fc1_flat,weights) + bias)
    #in case of over fit
    #keep_prob = tf.placeholder(tf.float32)
    fc1_drop = tf.nn.dropout(fc1, keep_prob=0.4)
    #print(fc1)

    #second full connected layer fc2
    weights_last_layer = _variables_on_cpu('fc_weight2', [1024, 2], tf.truncated_normal_initializer(0, 1e-4))
    bias = _variables_on_cpu('biases_fc2', [2], tf.constant_initializer(0.0))
    softmax_layer = tf.add(tf.matmul(fc1_drop, weights_last_layer), bias)
    #print(softmax_layer)

    return softmax_layer

def loss(logits, labels):
    #change the structure of label matrix to a shape suitable for neural network(one hot)
    sparse_labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, FLAGS.batch_size, 1), 1)
    #labels = tf.reshape(labels, [128,1])
    concated = tf.concat([indices, sparse_labels], 1)
    dense_labels = tf.sparse_to_dense(concated, [FLAGS.batch_size, 2], 1.0, 0.0)
    #Calculate the average cross entropy loss across the batch
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=dense_labels, logits=logits, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    #The total loss is defined as the cross entorpy loss plus all of the weight decay terms (L2 Loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_sumaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(decay=0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    return loss_averages_op

def train(total_loss, global_step):
    num_batches_per_epoch = image_train_size / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    #decay the learning rate exponetially based on the number of steps
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step=global_step,
                                    decay_steps=decay_steps, decay_rate=LEARNING_RATE_DECAY_FACTOR,staircase=True)
    loss_averages_op = _add_loss_sumaries(total_loss)
    #Compute gradients
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        grads = opt.compute_gradients(total_loss)

    #Apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import time
import model
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', "/Users/ComingWind/Documents/2017/ML/cat_and_dog_log/",'train dir')
tf.app.flags.DEFINE_integer('max_steps', 25000,'number of batches to run')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'whether to log device placement')

def train():

    with tf.Graph().as_default():
        global_setp = tf.Variable(0, trainable=False)

        #input images
        images, labels = model.distord_inputs()
        sesss = tf.InteractiveSession()
        print(labels)

        #Compute the model
        logits = model.model_build(images=images)

        #Calculate loss
        loss = model.loss(logits, labels)

        #Get train operation
        train_op = model.train(loss, global_setp)

        #Create a saver
        saver = tf.train.Saver(tf.global_variables())

        #Initialize
        init = tf.global_variables_initializer()

        #Start running all the operations on the Graph
        sess = tf.Session()
        sess.run(init)

        #Start the queue runners
        tf.train.start_queue_runners(sess=sess)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                examples_per_sec, sec_per_batch))

def main(argv=None):
    if gfile.Exists(FLAGS.train_dir):
        gfile.DeleteRecursively(FLAGS.train_dir)
    gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.python.platform import gfile

IMAGE_AMOUNTS_FOR_TRAINING = 25000
IMAGE_AMOUNTS_FOR_EVAL = 12500

CLASS_AMOUNT = 2

class Record(object):
    def __init__(self, height, width, depth, uint8image, label, key, value):
        self.height = height
        self.width = width
        self.depth = depth
        self.uint8image = uint8image
        self.label = label
        self.key = key
        self.value = value

def read_data(filename_sequence):

    label_bytes = 1
    height = 32
    width = 32
    depth = 3
    image_bytes = height * width * depth

    record_bytes = image_bytes + label_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_sequence)

    record_bytes = tf.decode_raw(value, tf.uint8)

    label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]), [depth, height, width])
    uint8image = tf.transpose(depth_major, [1, 2, 0])

    result = Record(height, width, depth, uint8image, label, key, value)

    return result

def _generate_images_and_label_batch(image, label, min_queue_examples, batch_size):
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch([image, label],
                                                 batch_size=batch_size
                                                 , capacity=min_queue_examples + 3 * batch_size,
                                                 min_after_dequeue=min_queue_examples
                                                 ,num_threads=num_preprocess_threads)


    return images, tf.reshape(label_batch, [batch_size])

def inputs(eval_data, data_dir, batch_size):

    if not eval_data:
        filenames = [os.path.join(data_dir, 'train_data%d' % i) for i in range(1, 6)]
    else:
        filenames = [os.path.join(data_dir, 'test_data.txt')]

    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_data(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           24, 24)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(IMAGE_AMOUNTS_FOR_TRAINING *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_images_and_label_batch(resized_image, read_input.label,
                                           min_queue_examples, batch_size)

def distorted_inputs(data_dir, batch_size):
    #filenames = [os.path.join(data_dir, 'train_data%d' % i)
    #             for i in xrange(1, 6)]
    filenames = [os.path.join(data_dir, 'train_data1')]
    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_data(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    distorted_image = tf.image.random_flip_left_right(reshaped_image)
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
    #float_image = tf.image.per_image_whitening(distorted_image)
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(IMAGE_AMOUNTS_FOR_TRAINING *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d cat and dog images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_images_and_label_batch(distorted_image, read_input.label,
                                           min_queue_examples, batch_size)

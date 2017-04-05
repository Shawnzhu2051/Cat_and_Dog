# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.python.platform

import tensorflow as tf
import cv2

#from tensorflow.models.image.cifar10 import cifar10_input
#from tensorflow.python.util import compat
import compat as cp

import handle_inputs

class inputtest(tf.test.TestCase):

    '''
    def _record(self, label, red, green, blue):
        image_size = 32 * 32
        record = bytes(bytearray([label] + [red] * image_size + [green] * image_size + [blue] * image_size))
        expected = [[[red, green, blue]] * 32] * 32
        return record, expected

    def xxxxx():
        labels = [0, 1, 0]
        records = [self._record(labels[0], 1, 128, 255),
                   self._record(labels[1], 2, 0, 1),
                   self._record(labels[2], 254, 255, 0)]
        contents = b"".join([record for record, _ in records])
        #contents = [record for record, _ in records]
        expected = [expected for _, expected in records]
        filename = os.path.join("/Users/ComingWind/Documents/2017/ML/", "testrecords")
        open(filename, "wb").write(contents)
        '''


    n = 1
    table1 = []
    table2 = []
    table3 = []
    table4 = []
    table5 = []

    def mybin(num):
        bstr = bin(num)
        bstr = '0' + bstr
        return bstr.replace('0b', '')


    def myhex(num):
        hstr = hex(num).replace('0x', '')
        if len(hstr) == 1:
            hstr = '0' + hstr
        return hstr


    def processing(n):
        sequence = str(n)
        string = 'cat.' + sequence

        image = cv2.imread('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/train/' + string + '.jpg')#按照文件名打开文件
        thirtytwo = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)#调整至32*32大小

        records = []
        label = [0,1]
        records.append(mybin(label[0]))

        for i in range(32):
            for j in range(0, 32, 2):
                temp1 = myhex(thirtytwo[i, j, 0])
                temp2 = myhex(thirtytwo[i, j + 1, 0])
                records.append(temp1 + temp2)
        for i in range(32):
            for j in range(0, 32, 2):
                temp1 = myhex(thirtytwo[i, j, 1])
                temp2 = myhex(thirtytwo[i, j + 1, 1])
                records.append(temp1 + temp2)
        for i in range(32):
            for j in range(0, 32, 2):
                temp1 = myhex(thirtytwo[i, j, 2])
                temp2 = myhex(thirtytwo[i, j + 1, 2])
                records.append(temp1 + temp2)
        contents = b"".join([record for record in records])
        with open('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/train_data1.txt', 'wb') as filename:
            filename.write(contents)

        with self.test_session() as sess:
            q = tf.FIFOQueue(99, [tf.string], shapes=())
            q.enqueue([filename]).run()
            q.close().run()
            result = handle_inputs.read_data(q)

            for i in range(3):
                key, label, uint8image = sess.run([result.key, result.label, result.uint8image])
                self.assertEqual("%s:%d" % (filename, i), cp.as_text(key))
                self.assertEqual(labels[i], label)
                self.assertAllEqual(expected[i], uint8image)
                #print(result)

            with self.assertRaises(tf.errors.OutOfRangeError):
                sess.run([result.key, result.uint8image])

if __name__ == "__main__":
    tf.test.main()
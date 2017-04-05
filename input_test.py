from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.python.platform

import tensorflow as tf

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
    '''

    def testSimple(self):
        '''
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
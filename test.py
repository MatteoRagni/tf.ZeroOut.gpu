#!/usr/bin/env python

import tensorflow as tf
zero_out_module = tf.load_op_library('zero_out.so')
with tf.Session(''):
  ret = zero_out_module.zero_out([[1, 2], [3, 4]]).eval()
print(ret)
exit(0)

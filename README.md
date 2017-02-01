# Zero Out GPU

## Description

A **CUDA** implementation of the example presented in [Tensorflow tutorials](https://www.tensorflow.org/how_tos/adding_an_op/#using_the_op_in_python) for custom ops.

> Sets all elements of the input tensor to 0, except for the very first one that is kept.

## Compile

To compile CPU version (the one proposed in the example):

```
make cpu
```

To compile the GPU version:

```
make gpu
```

## Usage

A simple Python test is presented in `test.py`:

```python
import tensorflow as tf

zero_out_module = tf.load_op_library('zero_out.so')

with tf.Session(''):
  ret = zero_out_module.zero_out([[1, 2], [3, 4]]).eval()

print(ret)
```

It prints:

```
[[1 0]
 [0 0]]
```

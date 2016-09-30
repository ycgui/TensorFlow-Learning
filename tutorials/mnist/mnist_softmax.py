# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

# Download and read MNIST data
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

# Build the entire computation graph before starting a session and launching the graph.
sess = tf.InteractiveSession()

# 1. Create the model

# placeholder: a value that we'll input when we ask TensorFlow to run a computation
# We represent this as a 2-D tensor of floating-point numbers, with a shape [None, 784]. (Here None means that a dimension can be of any length.)
x = tf.placeholder(tf.float32, [None, 784]) 
# A Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations. 
# It can be used and even modified by the computation. 
# For machine learning applications, one generally has the model parameters be Variables.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)


# 2. Define loss and optimizer

# Target y
y_ = tf.placeholder(tf.float32, [None, 10])
# cross entropy: h(y) = -sum(y_target * log(y_predicted))
# reduction_indices=[1]: use the second dimension of variables for computation
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# Optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# 3. Train

# Initialization
tf.initialize_all_variables().run()
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})


# 4. Test trained model

# argmax() is an extremely useful function which gives you the index of the highest entry in a tensor along some axis
# correct_prediction gives us a list of booleans
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# Cast booleans to floating point numbers and then take the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Report test accuracy
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

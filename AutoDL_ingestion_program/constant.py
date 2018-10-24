# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trivial example of learning algorithm.

"ConstantAlgorithm" is the simplest algorithm that returns the first output
seen on the training set.
"""

import tensorflow as tf
import algorithm


class Constant(algorithm.Algorithm):
  """Trivial algorithm returning the 1st training output for any test input."""

  def __init__(self, metadata):
    super(Constant, self).__init__(metadata)

  def train(self, dataset):
    """Keeps only the first example of the training set."""
    dataset_iterator = dataset.make_one_shot_iterator()
    # The next lines assume that
    # (a) get_next() returns a minibatch of examples
    # (b) each minibatch is a pair (inputs, outputs)
    # (c) the outputs has the same length as the inputs
    # We get the first minibatch by get_next,
    # then the output by [1], then the first example by [0].
    with tf.Session() as sess:
      self.first_example_output = sess.run(dataset_iterator.get_next()[1][0])
      # print("*"*50, "constant.py", dataset_iterator.get_next()[1][0].shape)

  def predict(self, *input_arg):
    return self.first_example_output

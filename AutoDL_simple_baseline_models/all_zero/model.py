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
# Modified by: Zhengying Liu, Isabelle Guyon

"""An example of code submission for the AutoDL challenge.

It implements 3 compulsory methods: __init__, train, and test.
model.py follows the template of the abstract class algorithm.py found
in folder AutoDL_ingestion_program/.

To create a valid submission, zip model.py together with other necessary files
such as Python modules/packages, pre-trained weights. The final zip file should
not exceed 300MB.
"""

import tensorflow as tf
import os
import numpy as np

# Import the challenge algorithm (model) API from algorithm.py
import algorithm

# Other useful modules
import datetime

class Model(algorithm.Algorithm):
  """Trivial example of valid model. Returns all-zero predictions."""

  def __init__(self, metadata):
    """
    Args:
      metadata: an AutoDLMetadata object. Its definition can be found in
          AutoDL_ingestion_program/dataset.py
    """
    super(Model, self).__init__(metadata)
    self.no_more_training = False

  def train(self, dataset, remaining_time_budget=None):
    """Train this algorithm on the tensorflow |dataset|.

    This method will be called REPEATEDLY during the whole training/predicting
    process. So your `train` method should be able to handle repeated calls and
    hopefully improve your model performance after each call.

    Args:
      dataset: a `tf.data.Dataset` object. Each of its examples is of the form
            (example, labels)
          where `example` is a dense 4-D Tensor of shape
            (sequence_size, row_count, col_count, num_channels)
          and `labels` is a 1-D Tensor of shape
            (output_dim,).
          Here `output_dim` represents number of classes of this
          multilabel classification task.

          IMPORTANT: some of the dimensions of `example` might be `None`,
          which means the shape on this dimension might be variable. In this
          case, some preprocessing technique should be applied in order to
          feed the training of a neural network. For example, if an image
          dataset has `example` of shape
            (1, None, None, 3)
          then the images in this datasets may have different sizes. On could
          apply resizing, cropping or padding in order to have a fixed size
          input tensor.

      remaining_time_budget: time remaining to execute train(). The method
          should keep track of its execution time to avoid exceeding its time
          budget. If remaining_time_budget is None, no time budget is imposed.
    """
    if self.no_more_training:
      return

    print_log("This basic sample code doesn't do any training, ",
              "but will show some information on the dataset:")
    iterator = dataset.make_one_shot_iterator()
    example, labels = iterator.get_next()
    sample_count = 0
    with tf.Session() as sess:
      while True:
        try:
          sess.run(labels)
          sample_count += 1
        except tf.errors.OutOfRangeError:
          break
    print_log("Number of training examples: {}".format(sample_count))
    print_log("Shape of example: {}".format(example.shape))
    print_log("Number of classes: {}".format(labels.shape[0]))
    pass

  def test(self, dataset, remaining_time_budget=None):
    """Make predictions on the test set `dataset` (which is different from that
    of the method `train`).

    Args:
      Same as that of `train` method, except that the `labels` will be empty
          since this time `dataset` is a test set.
    Returns:
      predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
          here `sample_count` is the number of examples in this dataset as test
          set and `output_dim` is the number of labels to be predicted. The
          values should be binary or in the interval [0,1].

          IMPORTANT: if returns `None`, this means that the algorithm
          chooses to stop training, and the whole train/test will stop. The
          performance of the last prediction will be used to compute area under
          learning curve.
    """
    if self.no_more_training:
      return None

    self.no_more_training = True
    sample_count = 0
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
      while True:
        try:
          sess.run(next_element)
          sample_count += 1
        except tf.errors.OutOfRangeError:
          break
    print_log("Number of test examples: {}".format(sample_count))
    output_dim = self.metadata_.get_output_size()
    predictions = np.zeros((sample_count, output_dim))
    return predictions

  ##############################################################################
  #### Above 3 methods (__init__, train, test) should always be implemented ####
  ##############################################################################

#### Can contain other functions too
def print_log(*content):
  """Logging function. (could've also used `import logging`.)"""
  now = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
  print("MODEL INFO: " + str(now)+ " ", end='')
  print(*content)

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

It implements 3 compulsory methods ('__init__', 'train' and 'test') and
an attribute 'done_training' for indicating if the model will not proceed more
training due to convergence or limited time budget.

To create a valid submission, zip model.py together with other necessary files
such as Python modules/packages, pre-trained weights. The final zip file should
not exceed 300MB.
"""

from tensorflow.python.client import device_lib
import logging
import numpy as np
import os
import sys
import tensorflow as tf

class Model(object):
  """Trivial example of valid model. Returns all-zero predictions."""

  def __init__(self, metadata):
    """
    Args:
      metadata: an AutoDLMetadata object. Its definition can be found in
          AutoDL_ingestion_program/dataset.py
    """
    self.done_training = False
    self.metadata = metadata
    # Show system info
    logger.info("System info ('uname -a'):")
    os.system('uname -a')
    # Show available devices
    local_device_protos = device_lib.list_local_devices()
    logger.info("Available local devices:\n{}".format(local_device_protos))
    # Show CUDA version
    logger.info("CUDA version:")
    os.system('nvcc --version')
    logger.info("Output of the command line 'nvclock':")
    os.system('nvclock')

  def train(self, dataset, remaining_time_budget=None):
    """Train this algorithm on the tensorflow |dataset|.

    This method will be called REPEATEDLY during the whole training/predicting
    process. So your `train` method should be able to handle repeated calls and
    hopefully improve your model performance after each call.

    ****************************************************************************
    ****************************************************************************
    IMPORTANT: the loop of calling `train` and `test` will only run if
        self.done_training = False
      (the corresponding code can be found in ingestion.py, search
      'M.done_training')
      Otherwise, the loop will go on until the time budget is used up. Please
      pay attention to set self.done_training = True when you think the model is
      converged or when there is not enough time for next round of training.
    ****************************************************************************
    ****************************************************************************

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
    logger.info("This basic sample code doesn't do any training, " +
              "but will show some information on the dataset:")
    X_train, Y_train = self.to_numpy(dataset, is_training=True)
    sample_count = len(X_train)
    if sample_count > 0:
      num_classes = len(Y_train[0])
    else: # normally this will never happen
      num_classes = None

    if has_regular_shape(dataset):
      X_train = np.array(X_train)
      Y_train = np.array(Y_train)
      print("X_train: {}".format(X_train))
      print("Y_train: {}".format(Y_train))

    logger.info("Number of training examples: {}".format(sample_count))
    logger.info("Number of classes: {}".format(num_classes))
    self.done_training = True

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
    """
    X_test, _ = self.to_numpy(dataset, is_training=False)
    sample_count = len(X_test)
    logger.info("Number of test examples: {}".format(sample_count))
    output_dim = self.metadata.get_output_size()
    predictions = np.zeros((sample_count, output_dim))
    return predictions

  ##############################################################################
  #### Above 3 methods (__init__, train, test) should always be implemented ####
  ##############################################################################

  def to_numpy(self, dataset, is_training):
    """Given the TF dataset received by `train` or `test` method, compute two
    lists of NumPy arrays: `X_train`, `Y_train` for `train` and `X_test`,
    `Y_test` for `test`. Although `Y_test` will always be an
    all-zero matrix, since the test labels are not revealed in `dataset`.

    The computed two lists will by memorized as object attribute:
      self.X_train
      self.Y_train
    or
      self.X_test
      self.Y_test
    according to `is_training`.

    WARNING: since this method will load all data in memory, it's possible to
      cause Out Of Memory (OOM) error, especially for large datasets (e.g.
      video/image datasets).

    Args:
      dataset: a `tf.data.Dataset` object, received by the method `self.train`
        or `self.test`.
      is_training: boolean, indicates whether it concerns the training set.
    Returns:
      two lists of NumPy arrays, for features and labels respectively. If the
        examples all have the same shape, they can be further converted to
        NumPy arrays by:
          X = np.array(X)
          Y = np.array(Y)
        And in this case, `X` will be of shape
          [num_examples, sequence_size, row_count, col_count, num_channels]
        and `Y` will be of shape
          [num_examples, num_classes]
    """
    if is_training:
      subset = 'train'
    else:
      subset = 'test'
    attr_X = 'X_{}'.format(subset)
    attr_Y = 'Y_{}'.format(subset)

    # Only iterate the TF dataset when it's not done yet
    if not (hasattr(self, attr_X) and hasattr(self, attr_Y)):
      iterator = dataset.make_one_shot_iterator()
      next_element = iterator.get_next()
      X = []
      Y = []
      with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        while True:
          try:
            example, labels = sess.run(next_element)
            X.append(example)
            Y.append(labels)
          except tf.errors.OutOfRangeError:
            break
      setattr(self, attr_X, X)
      setattr(self, attr_Y, Y)
    X = getattr(self, attr_X)
    Y = getattr(self, attr_Y)
    return X, Y

def has_regular_shape(dataset):
  """Check if the examples of a TF dataset has regular shape."""
  with tf.Graph().as_default():
    iterator = dataset.make_one_shot_iterator()
    example, labels = iterator.get_next()
    return all([x > 0 for x in example.shape])

def get_logger(verbosity_level):
  """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO score.py: <message>
  """
  logger = logging.getLogger(__file__)
  logging_level = getattr(logging, verbosity_level)
  logger.setLevel(logging_level)
  formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging_level)
  stdout_handler.setFormatter(formatter)
  stderr_handler = logging.StreamHandler(sys.stderr)
  stderr_handler.setLevel(logging.WARNING)
  stderr_handler.setFormatter(formatter)
  logger.addHandler(stdout_handler)
  logger.addHandler(stderr_handler)
  logger.propagate = False
  return logger

logger = get_logger('INFO')

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
such as Python modules/packages, pre-trained weights, etc. The final zip file
should not exceed 300MB.
"""

from sklearn.linear_model import LinearRegression
import logging
import numpy as np
import os
import sys
import tensorflow as tf
import time

np.random.seed(42)
tf.logging.set_verbosity(tf.logging.ERROR)

class Model(object):
  """Construct a model with 3D CNN for classification."""

  def __init__(self, metadata):
    """
    Args:
      metadata: an AutoDLMetadata object. Its definition can be found in
          AutoDL_ingestion_program/dataset.py
    """
    self.done_training = False
    self.metadata = metadata

    # Get the output dimension, i.e. number of classes
    self.output_dim = self.metadata.get_output_size()
    # Set batch size (for both training and testing)
    self.batch_size = 1 # necessary when the shape is variable

    # Attributes for preprocessing
    # self.default_image_size = (112,112)
    # self.default_num_frames = 10
    self.default_shuffle_buffer = 100
    self.meta_features = {}

    # Attributes for managing time budget
    # Cumulated number of training steps
    self.birthday = time.time()
    self.total_train_time = 0
    self.cumulated_num_steps = 0
    self.estimated_time_per_step = None
    self.total_test_time = 0
    self.cumulated_num_tests = 0
    self.estimated_time_test = None
    # Critical number for early stopping
    self.num_epochs_we_want_to_train = 40

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
    # Get number of steps to train according to some strategy
    steps_to_train = self.get_steps_to_train(remaining_time_budget)
    if not self.has_fixed_size and not self.meta_features:
      self.extract_meta_features(dataset)

    if not hasattr(self, 'classifier'):
      # Get model function from class method below
      model_fn = self.model_fn
      # Classifier using model_fn
      self.classifier = tf.estimator.Estimator(model_fn=model_fn)
      logger.info("Using temporary folder as model directory: {}"\
                  .format(self.classifier.model_dir))

    # Count examples on training set
    if not hasattr(self, 'num_examples_train'):
      logger.info("Counting number of examples on train set.")
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
      self.num_examples_train = sample_count
      logger.info("Finished counting. There are {} examples for training set."\
                  .format(sample_count))

    if steps_to_train <= 0:
      logger.info("Not enough time remaining for training. " +
            "Estimated time for training per step: {:.2f}, "\
            .format(self.estimated_time_per_step) +
            "but remaining time budget is: {:.2f}. "\
            .format(remaining_time_budget) +
            "Skipping...")
      self.done_training = True
    elif self.choose_to_stop_early():
      logger.info("The model chooses to stop further training because " +
                  "The preset maximum number of epochs for training is " +
                  "obtained: self.num_epochs_we_want_to_train = " +
                  str(self.num_epochs_we_want_to_train))
      self.done_training = True
    else:
      msg_est = ""
      if self.estimated_time_per_step:
        msg_est = "estimated time for this: {:.2f} sec."\
                  .format(steps_to_train * self.estimated_time_per_step)
      logger.info("Begin training for another {} steps...{}"\
                  .format(steps_to_train, msg_est))

      # Prepare input function for training
      train_input_fn = lambda: self.input_function(dataset, is_training=True)

      # Start training
      train_start = time.time()
      self.classifier.train(input_fn=train_input_fn, steps=steps_to_train)
      train_end = time.time()

      # Update for time budget managing
      train_duration = train_end - train_start
      self.total_train_time += train_duration
      self.cumulated_num_steps += steps_to_train
      self.estimated_time_per_step = self.total_train_time / self.cumulated_num_steps
      logger.info("{} steps trained. {:.2f} sec used. ".format(steps_to_train, train_duration) +\
            "Now total steps trained: {}. ".format(self.cumulated_num_steps) +\
            "Total time used for training: {:.2f} sec. ".format(self.total_train_time) +\
            "Current estimated time per step: {:.2e} sec.".format(self.estimated_time_per_step))

  def test(self, dataset, remaining_time_budget=None):
    """Test this algorithm on the tensorflow |dataset|.

    Args:
      Same as that of `train` method, except that the `labels` will be empty.
    Returns:
      predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
          here `sample_count` is the number of examples in this dataset as test
          set and `output_dim` is the number of labels to be predicted. The
          values should be binary or in the interval [0,1].
    """
    test_begin = time.time()
    logger.info("Begin testing... ")

    # Count examples on test set
    if not hasattr(self, 'num_examples_test'):
      logger.info("Counting number of examples on test set.")
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
      self.num_examples_test = sample_count
      logger.info("Finished counting. There are {} examples for test set."\
                  .format(sample_count))

    # Prepare input function for testing
    test_input_fn = lambda: self.input_function(dataset, is_training=False)

    # Start testing (i.e. making prediction on test set)
    test_results = self.classifier.predict(input_fn=test_input_fn)

    predictions = [x['probabilities'] for x in test_results]
    predictions = np.array(predictions)
    test_end = time.time()
    # Update some variables for time management
    test_duration = test_end - test_begin
    self.total_test_time += test_duration
    self.cumulated_num_tests += 1
    self.estimated_time_test = self.total_test_time / self.cumulated_num_tests
    logger.info("Successfully made one prediction. {:.2f} sec used. ".format(test_duration) +\
          "Total time used for testing: {:.2f} sec. ".format(self.total_test_time) +\
          "Current estimated time for test: {:.2e} sec.".format(self.estimated_time_test))
    return predictions

  ##############################################################################
  #### Above 3 methods (__init__, train, test) should always be implemented ####
  ##############################################################################

  # Model functions that contain info on neural network architectures
  # Several model functions are to be implemented, for different domains
  def model_fn(self, features, labels, mode):
    """Auto-Scaling 3D CNN model.

    For more information on how to write a model function, see:
      https://www.tensorflow.org/guide/custom_estimators#write_a_model_function
    """
    input_layer = features

    # Replace missing values by 0
    hidden_layer = tf.where(tf.is_nan(input_layer),
                           tf.zeros_like(input_layer), input_layer)

    if self.has_fixed_size:
      sequence_size = self.metadata.get_tensor_shape()[0]
      row_count = self.metadata.get_tensor_shape()[1]
      col_count = self.metadata.get_tensor_shape()[2]
    else:
      if not self.meta_features:
        self.extract_meta_features()
      sequence_size = self.meta_features['mean_sequence_size_train']
      row_count = self.meta_features['mean_row_count_train']
      col_count = self.meta_features['mean_col_count_train']
    # num_3dcnn_layers = get_num_3dcnn_layers(sequence_size,
    #                                         row_count,
    #                                         col_count)
    num_3dcnn_layers = 3
    logger.info("Constructing a CNN with {} 3D CNN layers..."\
                .format(num_3dcnn_layers))

    # Repeatedly apply 3D CNN, followed by 3D max pooling
    # Double the number of filters after each iteration
    num_filters = 16
    for _ in range(num_3dcnn_layers):
      shape = hidden_layer.shape
      kernel_size = []
      pool_size = []
      for i in range(1, 4):
        if shape[i] and shape[i] < 3 and shape[i] > 0:
          kernel_size.append(shape[i])
        else:
          kernel_size.append(3)
        if shape[i] and shape[i] == 1:
          pool_size.append(shape[i])
        else:
          pool_size.append(2)
      hidden_layer = tf.layers.conv3d(inputs=hidden_layer,
                                      filters=num_filters,
                                      kernel_size=kernel_size,
                                      padding='same')
      hidden_layer= tf.layers.max_pooling3d(inputs=hidden_layer,
                                            pool_size=pool_size,
                                            strides=pool_size,
                                            padding='valid',
                                            data_format='channels_last')
      num_filters *= 2

    hidden_layer = tf.reduce_mean(hidden_layer, axis=[1,2,3])
    hidden_layer = tf.layers.dense(inputs=hidden_layer,
                                   units=64, activation=tf.nn.relu)
    hidden_layer = tf.layers.dropout(
        inputs=hidden_layer, rate=0.15,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=hidden_layer, units=self.output_dim)
    sigmoid_tensor = tf.nn.sigmoid(logits, name="sigmoid_tensor")

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # "classes": binary_predictions,
      # Add `sigmoid_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": sigmoid_tensor
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # For multi-label classification, a correct loss is sigmoid cross entropy
    loss = sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer()
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    assert mode == tf.estimator.ModeKeys.EVAL
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

  def input_function(self, dataset, is_training):
    """Given `dataset` received by the method `self.train` or `self.test`,
    prepare input to feed to model function.

    For more information on how to write an input function, see:
      https://www.tensorflow.org/guide/custom_estimators#write_an_input_function
    """
    dataset = dataset.map(lambda *x: (x[0], x[1]))

    if is_training:
      # Shuffle input examples
      dataset = dataset.shuffle(buffer_size=self.default_shuffle_buffer)
      # Convert to RepeatDataset to train for several epochs
      dataset = dataset.repeat()

    # Set batch size
    dataset = dataset.batch(batch_size=self.batch_size)

    iterator = dataset.make_one_shot_iterator()
    example, labels = iterator.get_next()
    return example, labels

  @property
  def has_fixed_size(self):
    tensor_4d_shape = self.metadata.get_tensor_shape()
    return all([s > 0 for s in tensor_4d_shape])

  def extract_meta_features(self, dataset, subset='train'):
    logger.info("Begin extracting meta-features...")
    iterator = dataset.make_one_shot_iterator()
    example, labels = iterator.get_next()
    shapes = []
    sample_count = 0
    with tf.Session() as sess:
      while True:
        try:
          shape = sess.run(example).shape
          shapes.append(shape)
          sample_count += 1
          if sample_count % 100 == 0:
            logger.debug("{} samples read...".format(sample_count))
        except tf.errors.OutOfRangeError:
          break
    setattr(self, 'num_examples_{}'.format(subset), sample_count)
    shapes = np.array(shapes)
    for idx, dim in enumerate(['sequence_size', 'row_count', 'col_count']):
      vec = shapes[:, idx]
      for stat in ['mean', 'max', 'min']:
        key = '{}_{}_{}'.format(stat, dim, subset)
        self.meta_features[key] = getattr(np, stat)(vec)
    logger.info("Finished. Meta-features extracted: {}"\
                .format(self.meta_features))

  def get_steps_to_train(self, remaining_time_budget):
    """Get number of steps for training according to `remaining_time_budget`.

    The strategy is:
      1. If no training is done before, train for 10 steps (ten batches);
      2. Otherwise, estimate training time per step and time needed for test,
         then compare to remaining time budget to compute a potential maximum
         number of steps (max_steps) that can be trained within time budget;
      3. Choose a number (steps_to_train) between 0 and max_steps and train for
         this many steps. Double it each time.
    """
    if not remaining_time_budget: # This is never true in the competition anyway
      remaining_time_budget = 1200 # if no time limit is given, set to 20min

    if not self.estimated_time_per_step:
      steps_to_train = 10
    else:
      if self.estimated_time_test:
        tentative_estimated_time_test = self.estimated_time_test
      else:
        tentative_estimated_time_test = 50 # conservative estimation for test
      max_steps = int((remaining_time_budget - tentative_estimated_time_test) / self.estimated_time_per_step)
      max_steps = max(max_steps, 1)
      if self.cumulated_num_tests < np.log(max_steps) / np.log(2):
        steps_to_train = int(2 ** self.cumulated_num_tests) # Double steps_to_train after each test
      else:
        steps_to_train = 0
    return steps_to_train

  def age(self):
    return time.time() - self.birthday

  def choose_to_stop_early(self):
    """The criterion to stop further training (thus finish train/predict
    process).
    """
    batch_size = self.batch_size
    num_examples = self.num_examples_train
    num_epochs = self.cumulated_num_steps * batch_size / num_examples
    logger.info("Model already trained for {} epochs.".format(num_epochs))
    return num_epochs > self.num_epochs_we_want_to_train # Train for at least certain number of epochs then stop

def sigmoid_cross_entropy_with_logits(labels=None, logits=None):
  """Re-implementation of this function:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

  Let z = labels, x = logits, then return the sigmoid cross entropy
    max(x, 0) - x * z + log(1 + exp(-abs(x)))
  (Then sum over all classes.)
  """
  labels = tf.cast(labels, dtype=tf.float32)
  relu_logits = tf.nn.relu(logits)
  exp_logits = tf.exp(- tf.abs(logits))
  sigmoid_logits = tf.log(1 + exp_logits)
  element_wise_xent = relu_logits - labels * logits + sigmoid_logits
  return tf.reduce_sum(element_wise_xent)

def get_num_3dcnn_layers(sequence_size, row_count, col_count,
                         num_neurons=1000, num_filters=16):
  """Compute the number of 3D CNN layers one needs such that the number of
  neurons in the hidden layers is (strictly) smaller than `num_neurons`.

  Each 3D CNN layer is specified by the following code:
  ```
  kernel_size = [min(3, sequence_size), min(3, row_count), min(3, col_count)]
  hidden_layer = tf.layers.conv3d(inputs=hidden_layer,
                                  filters=num_filters,
                                  kernel_size=kernel_size,
                                  padding='same')
  pool_size = [min(2, sequence_size), min(2, row_count), min(2, col_count)]
  hidden_layer= tf.layers.max_pooling3d(inputs=hidden_layer,
                                        pool_size=pool_size,
                                        strides=pool_size,
                                        padding='valid',
                                        data_format='channels_last')
  ```
  """
  total_expo = np.log2(sequence_size * row_count * col_count * num_filters
                       / num_neurons)
  if total_expo < 0:
    return 0
  s_expo = int(np.log2(sequence_size))
  r_expo = int(np.log2(row_count))
  c_expo = int(np.log2(col_count))
  expos = sorted([s_expo, r_expo, c_expo])
  num_3dcnn_layers = 0
  for i in range(len(expos)):
    expo = expos[i]
    if i == 0:
      prev = 0
    else:
      prev = expos[i - 1]
    if total_expo <= (expo - prev) * (len(expos) - i):
      num_3dcnn_layers += int(total_expo // (len(expos) - i)) + 1
      break
    else:
      num_3dcnn_layers += expo - prev
      total_expo -= (expo - prev) * (len(expos) - i)
  return num_3dcnn_layers

def get_logger(verbosity_level):
  """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO model.py: <message>
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

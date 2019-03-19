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

To create a valid submission, zip model.py together with an empty
file called metadata (this just indicates your submission is a code submission
and has nothing to do with the dataset metadata.
"""

import tensorflow as tf
import os

# Import the challenge algorithm (model) API from algorithm.py
import algorithm

# Utility packages
import time
import datetime
import numpy as np
np.random.seed(42)

# Add path containing the repo tensorflow/models
# see https://github.com/tensorflow/models
import sys
MODELS_DIR = '../../../models'
if not os.path.isdir(MODELS_DIR):
  raise ValueError("You need to git clone the GitHub repo "
                   "at https://github.com/tensorflow/models, "
                   "at the parallel level of autodl repo.")
sys.path.append(MODELS_DIR)
# import models
# from models.official.resnet import resnet_model

class Model(algorithm.Algorithm):
  """Construct CNN for classification."""

  def __init__(self, metadata):
    super(Model, self).__init__(metadata)
    self.col_count, self.row_count = self.metadata_.get_matrix_size(0)
    self.sequence_size = self.metadata_.get_sequence_size()
    self.output_dim = self.metadata_.get_output_size()

    # Set batch size (for both training and testing)
    self.batch_size = 32

    # Get dataset name.
    self.dataset_name = self.metadata_.get_dataset_name()\
                          .split('/')[-2].split('.')[0]

    # Infer dataset domain and use corresponding model function
    self.domain = self.infer_domain()
    # Construct the neural network according to inferred domain
    model_fn = self.get_model_fn()

    # Directory to store checkpoints of model during training
    model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             os.pardir,
                             'checkpoints_' + self.dataset_name)

    # Classifier using model_fn
    # It'll be used for both training and testing
    self.classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir)

    # Attributes for managing time budget
    # Cumulated number of training steps
    self.birthday = time.time()
    self.total_train_time = 0
    self.cumulated_num_steps = 0
    self.estimated_time_per_step = None
    self.total_test_time = 0
    self.cumulated_num_tests = 0
    self.estimated_time_test = None
    self.done_training = False
    ################################################
    # Important critical number for early stopping #
    ################################################
    self.num_epochs_we_want_to_train = max(40, self.output_dim)
    # Depends on number of classes (output_dim)
    # see the function self.choose_to_stop_early() below for more details

  def train(self, dataset, remaining_time_budget=None):
    """Train this algorithm on the tensorflow |dataset|.

    This method will be called REPEATEDLY during the whole training/predicting
    process. So your `train` method should be able to handle repeated calls and
    hopefully improve your model performance after each call.

    Args:
      dataset: a `tf.data.Dataset` object. Each example is of the form
            (matrix_bundle_0, matrix_bundle_1, ..., matrix_bundle_(N-1), labels)
          where each matrix bundle is a tf.Tensor of shape
            (sequence_size, row_count, col_count).
          The variable `labels` is a tf.Tensor of shape
            (output_dim,)
          where `output_dim` represents number of classes of this
          multilabel classification task. For the first version of AutoDL
          challenge, the number of bundles `N` will be set to 1.

      remaining_time_budget: time remaining to execute train(). The method
          should keep track of its execution time to avoid exceeding its time
          budget. If remaining_time_budget is None, no time budget is imposed.
    """
    if self.done_training:
      return

    train_input_fn = self.get_input_fn(is_test=False)

    if not remaining_time_budget: # This is never true in the competition anyway
      remaining_time_budget = 1200 # if no time limit is given, set to 20min

    # The following snippet of code intends to do
    # 1. If no training is done before, train for 10 steps (ten batches);
    # 2. Otherwise, estimate training time per step and time needed for test,
    #    then compare to remaining time budget to compute a potential maximum
    #    number of steps (max_steps) that can be trained within time budget;
    # 3. Choose a number (steps_to_train) between 0 and max_steps and train for
    #    this many steps. Double it each time.
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
    if steps_to_train <= 0:
      print_log("Not enough time remaining for training. " +\
            "Estimated time for training per step: {:.2f}, ".format(self.estimated_time_per_step) +\
            "and for test: {}, ".format(tentative_estimated_time_test) +\
            "but remaining time budget is: {:.2f}. ".format(remaining_time_budget) +\
            "Skipping...")
      self.done_training = True
    else:
      msg_est = ""
      if self.estimated_time_per_step:
        msg_est = "estimated time for this: " +\
                  "{:.2f} sec.".format(steps_to_train * self.estimated_time_per_step)
      print_log("Begin training for another {} steps...{}".format(steps_to_train, msg_est))
      train_start = time.time()
      # Start training
      self.classifier.train(
        input_fn=lambda:train_input_fn(dataset),
        steps=steps_to_train)
      train_end = time.time()
      # Update for time budget managing
      train_duration = train_end - train_start
      self.total_train_time += train_duration
      self.cumulated_num_steps += steps_to_train
      self.estimated_time_per_step = self.total_train_time / self.cumulated_num_steps
      print_log("{} steps trained. {:.2f} sec used. ".format(steps_to_train, train_duration) +\
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
          IMPORTANT: if returns None, this means that the algorithm
          chooses to stop training, and the whole train/test will stop. The
          performance of the last prediction will be used to compute area under
          learning curve.
    """
    if self.done_training:
      return None

    test_input_fn = self.get_input_fn(is_test=True)

    # The following snippet of code intends to do:
    # 0. Use the function self.choose_to_stop_early() to decide if stop the whole
    #    train/predict process for next call
    # 1. If there is time budget limit, and some testing has already been done,
    #    but not enough remaining time for testing, then return None to stop
    # 2. Otherwise: make predictions normally, and update some
    #    variables for time management
    if self.choose_to_stop_early():
      print_log("Oops! Choose to stop early for next call!")
      self.done_training = True
    test_begin = time.time()
    if remaining_time_budget and self.estimated_time_test and\
        self.estimated_time_test > remaining_time_budget:
      print_log("Not enough time for test. " +\
            "Estimated time for test: {:.2e}, ".format(self.estimated_time_test) +\
            "But remaining time budget is: {:.2f}. ".format(remaining_time_budget) +\
            "Stop train/predict process by returning None.")
      return None
    msg_est = ""
    if self.estimated_time_test:
      msg_est = "estimated time: {:.2e} sec.".format(self.estimated_time_test)
    print_log("Begin testing...", msg_est)
    # Start testing (i.e. making prediction on test set)
    test_results = self.classifier.predict(input_fn=lambda:test_input_fn(dataset))
    predictions = [x['probabilities'] for x in test_results]
    has_same_length = (len({len(x) for x in predictions}) == 1)
    print_log("Asserting predictions have the same number of columns...")
    assert(has_same_length)
    predictions = np.array(predictions)
    test_end = time.time()
    # Update some variables for time management
    test_duration = test_end - test_begin
    self.total_test_time += test_duration
    self.cumulated_num_tests += 1
    self.estimated_time_test = self.total_test_time / self.cumulated_num_tests
    print_log("[+] Successfully made one prediction. {:.2f} sec used. ".format(test_duration) +\
          "Total time used for testing: {:.2f} sec. ".format(self.total_test_time) +\
          "Current estimated time for test: {:.2e} sec.".format(self.estimated_time_test))
    return predictions

  ##############################################################################
  #### Above 3 methods (__init__, train, test) should always be implemented ####
  ##############################################################################

  def get_input_fn(self, is_test):

    def input_fn(dataset):
      """For more details on how to write input function, please see:
      https://www.tensorflow.org/guide/custom_estimators#write_an_input_function
      """
      # Turn `features` in the tensor tuples
      #   (matrix_bundle_0,...,matrix_bundle_(N-1), labels)
      # to a dict. This example model only uses the first matrix bundle
      # (i.e. matrix_bundle_0) (see the documentation of train() function above
      # for the description of each example)
      dataset = dataset.map(lambda *x: (x[0], x[-1]))
      # Sample seveval frames from the video to represent it
      def sample_frames(video, num_frames):
        sequence_size = video.shape[0]
        frames = []
        for i in range(num_frames):
          random_index = np.random.randint(0, sequence_size)
          frames.append(video[random_index])
        res = tf.stack(frames)
        print(res.shape) # TODO
        return res
      num_frames = 5
      dataset = dataset.map(lambda x,y: (sample_frames(x,num_frames), y))
      # For training set, shuffle and repeat
      if not is_test:
        buffer_size = 10 * self.batch_size * self.output_dim
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.repeat()
      # Set batch size
      dataset = dataset.batch(batch_size=self.batch_size)
      iterator = dataset.make_one_shot_iterator()
      features, labels = iterator.get_next()
      return features, labels

    return input_fn


  def neural_network_architecture(self, input_layer, mode):
    """Construct a feed-forward neural network architecture according to
    self.domain.

    Args:
      input_layer: a tensor (placeholder) of shape
            [batch_size, sequence_size, row_count, col_count].
    Returns:
      logits: a tensor of shape
            [batch_size, output_dim]
          where output_dim is equal to the number of classes. The entries should
          be
    """
    # Get shape info from metadata
    col_count = self.col_count
    row_count = self.row_count
    sequence_size = self.sequence_size
    output_dim = self.output_dim

    ### Begin constructing neural networks for each domain ###
    ### Tabular ###
    if self.domain == 'tabular':
      hidden_layer = tf.layers.flatten(input_layer)
      hidden_layer = tf.layers.dense(inputs=hidden_layer, units=64,
                                     activation=tf.nn.relu)
      hidden_layer =\
        tf.layers.dropout(inputs=hidden_layer, rate=0.15,
                          training=mode == tf.estimator.ModeKeys.TRAIN)
      logits = tf.layers.dense(inputs=hidden_layer, units=output_dim)


    ### Video ###
    elif self.domain == 'video':
      hidden_layer = tf.layers.flatten(input_layer)
      logits = tf.layers.dense(inputs=hidden_layer, units=output_dim)


    ### Speech ###
    elif self.domain == 'speech':
      raise NotImplementedError("No method implemented for speech.")


    ### Text ###
    elif self.domain == 'text':
      raise NotImplementedError("No method implemented for text.")


    ### Image ###
    elif self.domain == 'image':
      raise NotImplementedError("No method implemented for image.")


    ### Others ### (not impossible in general)
    else:
      raise ValueError("Wrong domain value: {}. Should be".format(self.domain) +
                       " in ['tabular', 'video', 'speech', 'text', 'image'].")
    return logits

  def get_model_fn(self):
    """Return a model function with signature model_fn(features, labels, mode)
    using the function self.neural_network_architecture.
    """
    def model_fn(features, labels, mode):
      print_log("Constructing model function for {} dataset..."\
                .format(self.domain))

      # col_count = self.col_count
      # row_count = self.row_count
      # sequence_size = self.sequence_size
      # output_dim = self.output_dim
      print('features.shape:', features.shape)
      batch_size, sequence_size, row_count, col_count = features.shape

      # Input Layer
      input_layer = features
      input_layer = tf.reshape(input_layer,
                               [-1, sequence_size, row_count, col_count])

      ### The whole network architecture is constructed in this line ###
      logits = self.neural_network_architecture(input_layer, mode)
      ##################################################################

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
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)

      # Add evaluation metrics (for EVAL mode)
      assert mode == tf.estimator.ModeKeys.EVAL
      eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(
              labels=labels, predictions=predictions["classes"])}
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    return model_fn

  # Some helper functions
  def infer_domain(self):
    col_count, row_count = self.metadata_.get_matrix_size(0)
    sequence_size = self.metadata_.get_sequence_size()
    output_dim = self.metadata_.get_output_size()
    if sequence_size > 1:
      if col_count == 1 and row_count == 1:
        return "speech"
      elif col_count > 1 and row_count > 1:
        return "video"
      else:
        return 'text'
    else:
      if col_count > 1 and row_count > 1:
        return 'image'
      else:
        return 'tabular'

  def age(self):
    return time.time() - self.birthday

  def choose_to_stop_early(self):
    """The criterion to stop further training (thus finish train/predict
    process).
    """
    # return self.cumulated_num_tests > 10 # Limit to make 10 predictions
    # return np.random.rand() < self.early_stop_proba
    batch_size = 30 # See ingestion program: D_train.init(batch_size=30, repeat=True)
    num_examples = self.metadata_.size()
    num_epochs = self.cumulated_num_steps * batch_size / num_examples
    return num_epochs > self.num_epochs_we_want_to_train # Train for certain number of epochs then stop

### Other utility functions ###
def print_log(*content):
  """Logging function. (could've also used `import logging`.)"""
  now = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
  print("MODEL INFO: " + str(now)+ " ", end='')
  print(*content)

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

#CREATE CNN STRUCTURE
"""----------------------------------------------------------------------------------------------------------------------------------------------------------------"""
def alexnet_model_fn(features, labels, mode):

    """INPUT LAYER"""
    input_layer = tf.reshape(features["x"], [-1, FLAGS.image_width, FLAGS.image_height, FLAGS.image_channels], name="input_layer") #Alexnet uses 227x227x3 input layer. '-1' means pick batch size randomly
    #print(input_layer)

    """%FIRST CONVOLUTION BLOCK
        The first convolutional layer filters the 227×227×3 input image with
        96 kernels of size 11×11 with a stride of 4 pixels. Bias of 1."""
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=96, kernel_size=[11, 11], strides=4, padding="valid", activation=tf.nn.relu)
    lrn1 = tf.nn.lrn(input=conv1, depth_radius=5, bias=1.0, alpha=0.0001/5.0, beta=0.75); #Normalization layer
    pool1_conv1 = tf.layers.max_pooling2d(inputs=lrn1, pool_size=[3, 3], strides=2) #Max Pool Layer
    #print(pool1_conv1)


    """SECOND CONVOLUTION BLOCK
    Divide the 96 channel blob input from block one into 48 and process independently"""
    conv2 = tf.layers.conv2d(inputs=pool1_conv1, filters=256, kernel_size=[5, 5], strides=1, padding="same", activation=tf.nn.relu)
    lrn2 = tf.nn.lrn(input=conv2, depth_radius=5, bias=1.0, alpha=0.0001/5.0, beta=0.75); #Normalization layer
    pool2_conv2 = tf.layers.max_pooling2d(inputs=lrn2, pool_size=[3, 3], strides=2) #Max Pool Layer
    #print(pool2_conv2)

    """THIRD CONVOLUTION BLOCK
    Note that the third, fourth, and fifth convolution layers are connected to one
    another without any intervening pooling or normalization layers.
    The third convolutional layer has 384 kernels of size 3 × 3
    connected to the (normalized, pooled) outputs of the second convolutional layer"""
    conv3 = tf.layers.conv2d(inputs=pool2_conv2, filters=384, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu)
    #print(conv3)

    #FOURTH CONVOLUTION BLOCK
    """%The fourth convolutional layer has 384 kernels of size 3 × 3"""
    conv4 = tf.layers.conv2d(inputs=conv3, filters=384, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu)
    #print(conv4)

    #FIFTH CONVOLUTION BLOCK
    """%the fifth convolutional layer has 256 kernels of size 3 × 3"""
    conv5 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu)
    pool3_conv5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2, padding="valid") #Max Pool Layer
    #print(pool3_conv5)


    #FULLY CONNECTED LAYER 1
    """The fully-connected layers have 4096 neurons each"""
    pool3_conv5_flat = tf.reshape(pool3_conv5, [-1, 6* 6 * 256]) #output of conv block is 6x6x256 therefore, to connect it to a fully connected layer, we can flaten it out
    fc1 = tf.layers.dense(inputs=pool3_conv5_flat, units=4096, activation=tf.nn.relu)
    #fc1 = tf.layers.conv2d(inputs=pool3_conv5, filters=4096, kernel_size=[6, 6], strides=1, padding="valid", activation=tf.nn.relu) #representing the FCL using a convolution block (no need to do 'pool3_conv5_flat' above)
    #print(fc1)

    #FULLY CONNECTED LAYER 2
    """since the output from above is [1x1x4096]"""
    fc2 = tf.layers.dense(inputs=fc1, units=4096, activation=tf.nn.relu)
    #fc2 = tf.layers.conv2d(inputs=fc1, filters=4096, kernel_size=[1, 1], strides=1, padding="valid", activation=tf.nn.relu)
    #print(fc2)

    #FULLY CONNECTED LAYER 3
    """since the output from above is [1x1x4096]"""
    logits = tf.layers.dense(inputs=fc2, units=FLAGS.num_of_classes, name="logits_layer")
    #fc3 = tf.layers.conv2d(inputs=fc2, filters=43, kernel_size=[1, 1], strides=1, padding="valid")
    #logits = tf.layers.dense(inputs=fc3, units=FLAGS.num_of_classes) #converting the convolutional block (tf.layers.conv2d) to a dense layer (tf.layers.dense). Only needed if we had used tf.layers.conv2d to represent the FCLs
    #print(logits)

    #PASS OUTPUT OF LAST FC LAYER TO A SOFTMAX LAYER
    """convert these raw values into two different formats that our model function can return:
    The predicted class for each example: a digit from 1–43.
    The probabilities for each possible target class for each example
    tf.argmax(input=fc3, axis=1: Generate predictions from the 43 last filters returned from the fc3. Axis 1 will apply argmax to the rows
    tf.nn.softmax(logits, name="softmax_tensor"): Generate the probability distribution
    """
    predictions = {
      "classes": tf.argmax(input=logits, axis=1, name="classes_tensor"),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      }

    #Return result if we were in prediction mode and not training
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #CALCULATE OUR LOSS
    """For both training and evaluation, we need to define a loss function that measures how closely the
    model's predictions match the target classes. For multiclass classification, cross entropy is typically used as the loss metric."""
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=FLAGS.num_of_classes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    tf.summary.scalar('Loss Per Stride', loss) #Just to see loss values per epoch (testing tensor board)

    #CONFIGURE TRAINING
    """Since the loss of the CNN is the softmax cross-entropy of the fc3 layer
    and our labels. Let's configure our model to optimize this loss value during
    training. We'll use a learning rate of 0.001 and stochastic gradient descent
    as the optimization algorithm:"""
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step()) #global_Step needed for proper graph on tensor board
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00005) #Very small learning rate used. Training will be slower at converging by better
        #train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    #ADD EVALUATION METRICS
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
"""-----------------------------------------------------------------------------------------------------------------------------------------------------------------"""

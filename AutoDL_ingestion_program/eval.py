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

"""Evaluate a learning algorithm on AutoDl datasets."""

import tensorflow as tf
from tensorflow import app
import constant
import dataset

def run_eval(alg, data, split=0.01):
  """Returns the number of errors on test set and number of test samples."""
  data_size = data.get_metadata().size()
  # Training
  train_size = int(split * data_size)
  data_set = data.get_dataset()
  training_set = data_set.take(train_size)
  alg.train(training_set)

  # Testing
  test_set = data_set.skip(train_size)

  def compute_output(*arg):
    # arg[0] is the input
    return tf.map_fn(alg.predict, arg[0])

  output_set = test_set.map(compute_output)
  print("*"*50, "output_set.shape: ", type(output_set))
  print("*"*50, "compute_output.shape: ", compute_output.shape)

  def extract_target(*arg):
    return arg[1]

  target_set = test_set.map(extract_target)

  eval_set = tf.data.Dataset.zip((output_set, target_set))
  eval_iterator = eval_set.make_one_shot_iterator()

  # Compute loss on the eval/test set.
  total_loss = 0
  total_count = 0
  get_next = eval_iterator.get_next()
  with tf.Session() as sess:
    while True:
      try:
        pred, label = sess.run(get_next)
        loss = sess.run(
            tf.reduce_sum(
                tf.to_float(
                    tf.not_equal(tf.arg_max(label, 1), tf.arg_max(pred, 1)))))
        total_loss += loss
        total_count += pred.shape[0]
        print("Current loss: " + str(total_loss) + "/" + str(total_count) +
              "\tAccuracy: " +
              "{0:.2f}%".format((1 - total_loss/total_count)*100)
              )
      except tf.errors.OutOfRangeError:
        break
  return total_loss, total_count


def main(argv):
  del argv  # Unused.
  dataset_name = "mnist"
  data_set = dataset.AutoDLDataset(dataset_name)
  data_set.init()
  # alg = constant.Constant(data_set.get_metadata())
  import scikit
  alg = scikit.ScikitAlgorithmNeuralNetwork(data_set.get_metadata())
  total_loss, total_count = run_eval(alg, data_set)
  print("Classification loss for " + dataset_name + ": " +
        str(total_loss * 1.0 / total_count))


if __name__ == "__main__":
  app.run(main)

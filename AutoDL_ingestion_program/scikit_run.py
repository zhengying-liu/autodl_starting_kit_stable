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

"""Tests for google3.experimental.autodl.tensorflow.algorithm."""

import sys
import tensorflow as tf
import scikit
import dataset

all_algorithms = []
all_algorithms.append(scikit.ScikitAlgorithmNeuralNetwork)
all_algorithms.append(scikit.ScikitAlgorithmBigNeuralNetwork)
all_algorithms.append(scikit.ScikitAlgorithmNN)
all_algorithms.append(scikit.ScikitAlgorithm3NN)
all_algorithms.append(scikit.ScikitAlgorithm5NN)
all_algorithms.append(scikit.ScikitAlgorithmLDA)
all_algorithms.append(scikit.ScikitAlgorithmQDA)


def RunAlgorithm(alg):
    # We loop over all registered algorithms.
    full_dataset = dataset.AutoDLDataset("mnist")
    full_dataset.init()
    train_dataset = full_dataset.get_dataset().take(10000)
    learning_alg = alg(full_dataset.get_metadata())
    # "dataset" is used as a learning set.
    learning_alg.train_by_time(train_dataset, max_time=10)
    print("Learning done with algorithm " + str(learning_alg))
    sys.stdout.flush()

    # We create the test dataset, includes the training set here.
    test_dataset = dataset.AutoDLDataset("mnist")
    test_dataset.init()
    iterator = test_dataset.get_dataset().take(10).make_one_shot_iterator()
    next_element = iterator.get_next()

    sess = tf.Session()
    idx = 0
    for _ in range(1):  # Loop over minibatches.
      examples = sess.run(next_element)
      for example in examples[0]:  # Loop over examples in the minibatch.
        idx += 1
        print("Output = " + str(learning_alg.predict(example)))


if __name__ == "__main__":
  for a in all_algorithms:
    RunAlgorithm(a)

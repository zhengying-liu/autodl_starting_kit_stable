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

"""Class for supervised machine learning algorithms for the autodl project.

A bit less naive than algorithms in constant.py.

Uses a small part of the dataset so that nothing breaks up when the
dataset is large.
"""

import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
import algorithm


class ScikitAlgorithm(algorithm.Algorithm):
  """Wrapper on scikit, adapted to autodl/dataset.py datasets."""

  def __init__(self, metadata):
    super(ScikitAlgorithm, self).__init__(metadata)

  def create_scikit_model(self):
    raise NotImplementedError("ScikitAlgorithm class is abstract.")

  def learn(self, inputs, outputs):
    self.create_scikit_model()
    self.my_model.fit(inputs, outputs)

  def flatten(self, example):
    try:
      return [float(example)]
    except TypeError:  # This is a complex type.
      flat = []
      for ex in list(example):
        flat += self.flatten(ex)
      return flat

  def train_by_time(self, dataset, max_time):
    start_time = time.time()
    max_num_minibatches = 10
    for _ in range(100):
      self.train(dataset, max_num_minibatches)
      elapsed_time = time.time() - start_time
      if elapsed_time > max_time / 2:
        break
      max_num_minibatches *= 2

  def train(self, dataset, max_num_minibatches=30):
    """Train this algorithm on |dataset|."""
    dataset_iterator = dataset.make_one_shot_iterator()

    inputs = []
    outputs = []

    with tf.Session() as sess:
      iterator = dataset_iterator.get_next()
      # We add a limited number of examples.
      for _ in range(max_num_minibatches):
        try:
          input_minibatch, output_minibatch = sess.run(iterator)
        except ValueError:
          break
        for inp in input_minibatch:
          # We flatten examples.
          inputs.append(self.flatten(inp))
        for out in output_minibatch:
          outputs.append(list(out))

    # We call scikit-learn.
    print("*"*50, input_minibatch.shape)
    print("*"*50, len(inputs))
    self.learn(inputs, outputs)

  def predict(self, *input_arg):
    """input_arg should correspond to a single sample."""
    prediction = self.my_model.predict([self.flatten(input_arg)])
    assert len(prediction) == 1
    return prediction[0]


class ScikitAlgorithmNeuralNetwork(ScikitAlgorithm):
  """Scikit bigger neural network."""

  def create_scikit_model(self):
    self.my_model = MLPRegressor()


class ScikitAlgorithmBigNeuralNetwork(ScikitAlgorithm):
  """Scikit bigger neural network."""

  def create_scikit_model(self):
    self.my_model = MLPRegressor(hidden_layer_sizes=(100, 100))


class ScikitAlgorithmNN(ScikitAlgorithm):
  """Scikit nearest neighbor."""

  def create_scikit_model(self):
    self.my_model = KNeighborsRegressor(n_neighbors=1)


class ScikitAlgorithm3NN(ScikitAlgorithm):
  """Scikit 3 nearest neighbors."""

  def create_scikit_model(self):
    self.my_model = KNeighborsRegressor(n_neighbors=3)


class ScikitAlgorithm5NN(ScikitAlgorithm):
  """Scikit 5 nearest neighbors."""

  def create_scikit_model(self):
    self.my_model = KNeighborsRegressor(n_neighbors=5)


class ScikitAlgorithmLDA(ScikitAlgorithm):
  """Scikit linear discriminant analysis."""

  def create_scikit_model(self, num_classes):
    self.my_model = []
    for _ in range(num_classes):
      self.my_model.append(LinearDiscriminantAnalysis())

  def learn(self, inputs, outputs):
    dimension = len(outputs[0])
    self.create_scikit_model(dimension)
    for i in range(dimension):
      self.my_model[i].fit(inputs, [output[i] for output in outputs])

  def predict(self, *input_arg):
    return [m.predict([self.flatten(input_arg)])[0] for m in self.my_model]


class ScikitAlgorithmQDA(ScikitAlgorithmLDA):
  """Scikit quadratic discriminant analysis."""

  def create_scikit_model(self, num_classes):
    self.my_model = []
    for _ in range(num_classes):
      self.my_model.append(QuadraticDiscriminantAnalysis())

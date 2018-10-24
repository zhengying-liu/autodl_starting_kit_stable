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
import numpy as np

# Import the challenge algorithm (model) API from algorithm.py
import algorithm

class Model(algorithm.Algorithm):
  """Return all-zero predictions."""

  def __init__(self, metadata):
    super(Model, self).__init__(metadata)
    self.no_more_training = False

  def train(self, dataset, remaining_time_budget=None):
    """No training."""
    pass

  def test(self, dataset, remaining_time_budget=None):
    """Return all zeros"""
    if self.no_more_training:
      return None
    else:
      self.no_more_training = True
      sample_count = self.metadata_.size() # sample count for test set (see ingestion program)
      output_dim = self.metadata_.get_output_size()
      predictions = np.zeros((sample_count, output_dim))
      return predictions

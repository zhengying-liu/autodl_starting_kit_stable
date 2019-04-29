################################################################################
# Name:         Ingestion Program
# Author:       Zhengying Liu, Isabelle Guyon, Adrien Pavao, Zhen Xu
# Update time:  Apr 29 2019
# Usage: python ingestion.py input_dir output_dir ingestion_program_dir submission_program_dir
#                            data      result     ingestion             code of participants

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.

VERSION = 'v20190429'
DESCRIPTION =\
"""This is the "ingestion program" written by the organizers. It takes the
code written by participants (with `model.py`) and one dataset as input,
run the code on the dataset and produce predictions on test set. For more
information on the code/directory structure, please see comments in this
code (ingestion.py) and the README file of the starting kit.
Previous updates:
20190429: [ZY] Remove useless code block; better code layout.
20190425: [ZY] Check prediction shape.
20190424: [ZY] Use logging instead of logger; remove start.txt checking;
20190419: [ZY] Try-except clause for training process;
          always terminates successfully.
"""
# The input directory input_dir (e.g. AutoDL_sample_data/) contains one dataset
# folder (e.g. adult.data/) with the training set (train/)  and test set (test/),
# each containing an some tfrecords data with a `metadata.textproto` file of
# metadata on the dataset. So one AutoDL dataset will look like
#
#   adult.data
#   ├── test
#   │   ├── metadata.textproto
#   │   └── sample-adult-test.tfrecord
#   └── train
#       ├── metadata.textproto
#       └── sample-adult-train.tfrecord
#
# The output directory output_dir (e.g. AutoDL_sample_result_submission/)
# will receive all predictions made during the whole train/predict process
# (thus this directory is updated when a new prediction is made):
# 	adult.predict_0
# 	adult.predict_1
# 	adult.predict_2
#        ...
# after ingestion has finished, a file duration.txt will be written, containing
# info on the duration ingestion used. This file is also used as a signal
# for scoring program showing that ingestion has terminated.
#
# The code directory submission_program_dir (e.g. AutoDL_sample_code_submission/)
# should contain your code submission model.py (and possibly other functions
# it depends upon).
#
# We implemented several classes:
# 1) DATA LOADING:
#    ------------
# dataset.py
# dataset.AutoDLMetadata: Read metadata in metadata.textproto
# dataset.AutoDLDataset: Read data and give tf.data.Dataset
# 2) LEARNING MACHINE:
#    ----------------
# model.py
# model.Model.train
# model.Model.test
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# UNIVERSITE PARIS SUD, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL UNIVERSITE PARIS SUD AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.
#
# Main contributors: Isabelle Guyon and Zhengying Liu

# =========================== BEGIN OPTIONS ==============================

# Verbosity level of logging:
##############
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
verbosity_level = 'INFO'

# Time budget
#############
# Maximum time of training in seconds PER DATASET (there may be several datasets).
# The code should keep track of time spent and NOT exceed the time limit
time_budget = 7200

# Some common useful packages
from os import getcwd as pwd
from os.path import join
from sys import argv, path
import datetime
import glob
import logging
import numpy as np
import os
import sys
import time

# Set logging format to something like:
# 2019-04-25 12:52:51 INFO score.py: <message>
logging.basicConfig(
    level=getattr(logging, verbosity_level),
    format='%(asctime)s %(levelname)s %(filename)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def _HERE(*args):
  """Helper function for getting the current directory of this script."""
  h = os.path.dirname(os.path.realpath(__file__))
  return os.path.abspath(os.path.join(h, *args))

# Default I/O directories:
# root_dir is the parent directory of the folder "AutoDL_ingestion_program"
root_dir = _HERE(os.pardir)
default_input_dir = join(root_dir, "AutoDL_sample_data")
default_output_dir = join(root_dir, "AutoDL_sample_result_submission")
default_program_dir = join(root_dir, "AutoDL_ingestion_program")
default_submission_dir = join(root_dir, "AutoDL_sample_code_submission")

# =========================== BEGIN PROGRAM ================================

if __name__=="__main__":
    overall_start = time.time()         # <== Mark starting time
    the_date = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")

    #### Check whether everything went well
    ingestion_success = True

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
        program_dir= default_program_dir
        submission_dir= default_submission_dir
        score_dir = join(root_dir, "AutoDL_scoring_output")
    elif len(argv)==2: # the case for indicating special input_dir
        input_dir = argv[1]
        output_dir = default_output_dir
        program_dir= default_program_dir
        submission_dir= default_submission_dir
        score_dir = join(root_dir, "AutoDL_scoring_output")
    elif len(argv)==3: # the case for indicating special input_dir and submission_dir. The case for run_local_test.py
        input_dir = argv[1]
        output_dir = default_output_dir
        program_dir= default_program_dir
        submission_dir= argv[2]
        score_dir = join(root_dir, "AutoDL_scoring_output")
    else: # the case on CodaLab platform
        input_dir = os.path.abspath(os.path.join(argv[1], '../input_data'))
        output_dir = os.path.abspath(os.path.join(argv[1], 'res'))
        program_dir = os.path.abspath(argv[3])
        submission_dir = os.path.abspath(os.path.join(argv[4], '../submission'))
        score_dir = os.path.abspath(os.path.join(argv[4], '../output'))

    logging.debug("sys.argv = " + str(sys.argv))
    logging.debug("Using input_dir: " + input_dir)
    logging.debug("Using output_dir: " + output_dir)
    logging.debug("Using program_dir: " + program_dir)
    logging.debug("Using submission_dir: " + submission_dir)

	  # Our libraries
    path.append(program_dir)
    path.append(submission_dir)
    #IG: to allow submitting the starting kit as sample submission
    path.append(submission_dir + '/AutoDL_sample_code_submission')
    import data_io
    from dataset import AutoDLDataset # THE class of AutoDL datasets

    data_io.mkdir(output_dir)

    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(input_dir)
    #### Delete zip files and metadata file
    datanames = [x for x in datanames if x.endswith('.data')]

    if len(datanames) != 1:
      raise ValueError("Multiple (or zero) datasets found in dataset_dir={}!\n"\
                       .format(input_dir) +
                       "Please put only ONE dataset under dataset_dir.")

    basename = datanames[0]

    logging.info("========== Ingestion program version " + str(VERSION) +
                 " ==========")
    logging.info("************************************************")
    logging.info("******** Processing dataset " + basename[:-5].capitalize() +
                 " ********")
    logging.info("************************************************")

    logging.debug("Version: {}. Description: {}".format(VERSION, DESCRIPTION))

    ##### Begin creating training set and test set #####
    logging.info("Reading training set and test set...")
    D_train = AutoDLDataset(os.path.join(input_dir, basename, "train"))
    D_test = AutoDLDataset(os.path.join(input_dir, basename, "test"))
    ##### End creating training set and test set #####

    ## Get correct prediction shape
    num_examples_test = D_test.get_metadata().size()
    output_dim = D_test.get_metadata().get_output_size()
    correct_prediction_shape = (num_examples_test, output_dim)

    try:
      # ========= Creating a model
      from model import Model # in participants' model.py
      ##### Begin creating model #####
      logging.info("Creating model...")
      M = Model(D_train.get_metadata()) # The metadata of D_train and D_test only differ in sample_count
      ###### End creating model ######

      # Keeping track of how many predictions are made
      prediction_order_number = 0

      # Start the CORE PART: train/predict process
      start = time.time()
      while(True):
        remaining_time_budget = start + time_budget - time.time()
        # Train the model
        logging.info("Beging training the model...")
        M.train(D_train.get_dataset(),
                remaining_time_budget=remaining_time_budget)
        logging.info("Done training the model.")
        remaining_time_budget = start + time_budget - time.time()
        # Make predictions using the trained model
        logging.info("Begin testing the model by making predictions " +
                     "on test set...")
        Y_pred = M.test(D_test.get_dataset(),
                        remaining_time_budget=remaining_time_budget)
        logging.info("Done making predictions.")
        if Y_pred is None: # Stop train/predict process if Y_pred is None
          break
        else: # Check if the prediction has good shape
          prediction_shape = tuple(Y_pred.shape)
          if prediction_shape != correct_prediction_shape:
            raise ValueError("Bad prediction shape! Expected {} but got {}."\
                          .format(correct_prediction_shape, prediction_shape))
        # Prediction files: adult.predict_0, adult.predict_1, ...
        filename_test = basename[:-5] + '.predict_' +\
          str(prediction_order_number)
        # Write predictions to output_dir
        data_io.write(os.path.join(output_dir,filename_test), Y_pred)
        prediction_order_number += 1
        logging.info("[+] Prediction success, time spent so far {0:.2f} sec"\
                     .format(time.time() - start))
        remaining_time_budget = start + time_budget - time.time()
        logging.info( "[+] Time left {0:.2f} sec".format(remaining_time_budget))
        if remaining_time_budget<=0:
          break
    except Exception as e:
      ingestion_success = False
      logging.info("Failed to run ingestion.")
      logging.error("Encountered exception:\n" + str(e), exc_info=True)

    # Finishing ingestion program
    overall_time_spent = time.time() - overall_start

    # Write overall_time_spent to a duration.txt file
    duration_filename =  'duration.txt'
    with open(os.path.join(output_dir, duration_filename), 'w') as f:
      f.write('Duration: ' + str(overall_time_spent) + '\n')
      f.write('Success: ' + str(int(ingestion_success)) + '\n')
      logging.info("Successfully write duration to {}.".format(duration_filename))
      if ingestion_success:
          logging.info("[+] Done")
          logging.info("[+] Overall time spent %5.2f sec " % overall_time_spent)
      else:
          logging.info("[-] Done, but encountered some errors during ingestion")
          logging.info("[-] Overall time spent %5.2f sec " % overall_time_spent)

    # Copy all files in output_dir to score_dir
    os.system("cp -R {} {}".format(os.path.join(output_dir, '*'), score_dir))

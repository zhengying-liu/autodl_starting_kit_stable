#!/usr/bin/env python

# Usage: python ingestion.py input_dir output_dir ingestion_program_dir submission_program_dir
#                            data      result     ingestion             code of participants

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
#
# This is the "ingestion program" written by the organizers.
# This program also runs on the challenge platform to test your code.
#
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
# Verbose mode:
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True # outputs messages to stdout and stderr for debug purposes

# Debug level:
##############
# 0: run the code normally, using the time budget of the tasks
# 1: run the code normally, but limits the time to max_time
# 2: run everything, but do not train, generate random outputs in max_time
# 3: stop before the loop on datasets
# 4: just list the directories and program version
debug_mode = 0

# Time budget
#############
# Maximum time of training in seconds PER DATASET (there may be several datasets).
# The code should keep track of time spent and NOT exceed the time limit
# in the dataset "info" file, stored in D.info['time_budget'], see code below.
# If debug >=1, you can decrease the maximum time (in sec) with this variable:
max_time = 300

# Maximum number of cycles, number of samples, and estimators
#############################################################
# Your training algorithm may be fast, so you may want to limit anyways the
# number of points on your learning curve (this is on a log scale, so each
# point uses twice as many time than the previous one.)
# The original code was modified to do only a small "time probing" followed
# by one single cycle. We can now also give a maximum number of estimators
# (base learners).
max_cycle = 1
max_estimators = 1000
max_samples = float('Inf')

# Redirect stardant output to live results page (detailed_results.html)
# to have live output for debugging
REDIRECT_STDOUT = False

# I/O defaults
##############
# If true, the previous output directory is not overwritten, it changes name
save_previous_results = False

# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
import os
from os import getcwd as pwd
from os.path import join
import shutil # for deleting a whole directory
from functools import partial
import tensorflow as tf

def _HERE(*args):
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(h, *args)

# Default I/O directories:
# root_dir is the parent directory of the folder "AutoDL_ingestion_program"
root_dir = os.path.abspath(os.path.join(_HERE(), os.pardir))
default_input_dir = join(root_dir, "AutoDL_sample_data")
default_output_dir = join(root_dir, "AutoDL_sample_result_submission")
default_program_dir = join(root_dir, "AutoDL_ingestion_program")
default_submission_dir = join(root_dir, "AutoDL_sample_code_submission")


# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================

# Version of the sample code
version = 1

# General purpose functions
import time
import numpy as np
overall_start = time.time()         # <== Mark starting time
import os
import sys
from sys import argv, path
import datetime
the_date = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")

def print_log(*content):
  """Logging function. (could've also used `import logging`.)"""
  if verbose:
    now = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
    print("INGESTION INFO: " + str(now)+ " ", end='')
    print(*content)

def clean_last_output(output_dir):
  """Clean existing output_dir of possible last execution.

  This includes directories such as:
    AutoDL_sample_result_submission/
    AutoDL_scoring_output/
    checkpoints_adult/
    /tmp/checkpoints_adult/
    ...

  Using this function, the user doesn't need to worry about the generated
  folders of last execution.
  """
  if os.path.isdir(output_dir):
    if verbose:
      print_log("Cleaning existing output_dir: {}".format(output_dir))
    shutil.rmtree(output_dir)
  model_dir = os.path.join(os.path.dirname(os.path.realpath(model.__file__)),
                           os.pardir)
  for parent_dir in ['/tmp/', os.path.join(output_dir, os.pardir), model_dir]:
    # Clean existing checkpoints
    checkpoints_glob =\
      os.path.abspath(os.path.join(parent_dir, 'checkpoints*'))
    checkpoints_dirs = tf.gfile.Glob(checkpoints_glob)
    for checkpoints_dir in checkpoints_dirs:
      if verbose:
        print_log("Cleaning existing checkpoints_dir: {}"\
                  .format(checkpoints_dir))
      shutil.rmtree(checkpoints_dir)

def get_time_budget(autodl_dataset):
  """Time budget for a given AutoDLDataset."""
  # TODO: decision to make on time budget.
  # Now it's 2 hours for any dataset (to be discussed).
  return 7200

# =========================== BEGIN PROGRAM ================================

if __name__=="__main__" and debug_mode<4:
    #### Check whether everything went well (no time exceeded)
    execution_success = True

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
    elif len(argv)==3: # the case for indicating special input_dir and submission_dir
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

    # Redirect standard output to have live debugging info (esp. on CodaLab)
    if REDIRECT_STDOUT:
        sys.stdout = open(os.path.join(score_dir, 'detailed_results.html'), 'a')
        print = partial(print, flush=True)

	# Our libraries
    path.append(program_dir)
    path.append(submission_dir)
    #IG: to allow submitting the starting kit as sample submission
    path.append(submission_dir + '/AutoDL_sample_code_submission')
    import data_io
    from data_io import vprint
    import model # participants' model.py
    from model import Model
    from dataset import AutoDLDataset # THE class of AutoDL datasets

    # Clear potentiablly results of previous execution (for local run)
    clean_last_output(output_dir)

    if debug_mode >= 4: # Show library version and directory structure
        data_io.show_dir(".")

    # Move old results and create a new output directory (useful if you run locally)
    if save_previous_results:
        data_io.mvdir(output_dir, output_dir+'_'+the_date)
    data_io.mkdir(output_dir)

    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(input_dir)
    #### Delete zip files and metadata file
    datanames = [x for x in datanames if x.endswith('.data')]

    #### DEBUG MODE: Show dataset list and STOP
    if debug_mode>=3:
        data_io.show_version()
        data_io.show_io(input_dir, output_dir)
        print_log('****** Ingestion program version ' + str(version) + ' ******\n\n' + '========== DATASETS ==========\n')
        data_io.write_list(datanames)
        datanames = [] # Do not proceed with learning and testing

    #### MAIN LOOP OVER DATASETS:
    overall_time_budget = 0
    time_left_over = 0

    # Loop over datasets (if several)
    # For AutoDL challenge, there is only 1 dataset for each track, so this loop
    # can actually be ignored. Here basename is e.g. 'adult.data'
    for i, basename in enumerate(datanames):

        print_log("========== Ingestion program version " + str(version) + " ==========")
        print_log("************************************************")
        print_log("******** Processing dataset " + basename[:-5].capitalize() + " ********")
        print_log("************************************************")

        # ======== Learning on a time budget:
        # Keep track of time not to exceed your time budget. Time spent to inventory data neglected.
        start = time.time()

        # ======== Creating a data object with data, informations about it
        print_log("Reading training set and test set...")

        ##### Begin creating training set and test set #####
        D_train = AutoDLDataset(os.path.join(input_dir, basename, "train"))
        D_test = AutoDLDataset(os.path.join(input_dir, basename, "test"))
        ##### End creating training set and test set #####

        # ======== Keep track of time
        if debug_mode<1:
            time_budget = get_time_budget(D_train)        # <== HERE IS THE TIME BUDGET!
        else:
            time_budget = max_time

        # ========= Creating a model
        print_log("Creating model...")
        ##### Begin creating model #####
        M = Model(D_train.get_metadata()) # The metadata of D_train and D_test only differ in sample_count
        ###### End creating model ######

        # Keeping track of how many predictions are made
        prediction_order_number = 0

        # Start the CORE PART: train/predict process
        start = time.time()
        while(True):
          remaining_time_budget = start + time_budget - time.time()
          print_log("Training the model...")
          # Train the model
          M.train(D_train.get_dataset(),
                  remaining_time_budget=remaining_time_budget)
          remaining_time_budget = start + time_budget - time.time()
          # Make predictions using the trained model
          Y_test = M.test(D_test.get_dataset(),
                          remaining_time_budget=remaining_time_budget)
          if Y_test is None: # Stop train/predict process if Y_test is None
            break
          # Prediction files: adult.predict_0, adult.predict_1, ...
          filename_test = basename[:-5] + '.predict_' +\
            str(prediction_order_number)
          # Write predictions to output_dir
          data_io.write(os.path.join(output_dir,filename_test), Y_test)
          prediction_order_number += 1
          print_log("[+] Prediction success, time spent so far %5.2f sec" % (time.time() - start))
          remaining_time_budget = start + time_budget - time.time()
          print_log( "[+] Time left %5.2f sec" % remaining_time_budget)
          if remaining_time_budget<=0:
            break

    # Finishing ingestion program
    overall_time_spent = time.time() - overall_start
    # Write overall_time_spent to a duration.txt file
    duration_filename =  'duration.txt'
    with open(os.path.join(output_dir, duration_filename), 'w') as f:
      f.write(str(overall_time_spent))
      if verbose:
          print_log("Successfully write duration to {}.".format(duration_filename))
    if execution_success:
        print_log("[+] Done")
        print_log("[+] Overall time spent %5.2f sec " % overall_time_spent)
    else:
        print_log("[-] Done, but some tasks aborted because time limit exceeded")
        print_log("[-] Overall time spent %5.2f sec " % overall_time_spent)

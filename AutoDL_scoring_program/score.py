################################################################################
# Name:         Scoring Program
# Author:       Zhengying Liu, Isabelle Guyon, Adrien Pavao, Zhen Xu
# Update time:  Apr 29 2019
# Usage: 		python score.py input_dir output_dir
#           input_dir contains two subdirectories 'res' and 'ref'
#                   'ref' contains e.g. adult.solution
#                   'res' contains e.g. duration.txt, adult.predict_0, adult.predict_1, etc.
#           output_dir should contain scores.txt, detailed_results.html

VERSION = 'v20190429'
DESCRIPTION =\
"""This is the scoring program for AutoDL challenge. It takes the predictions
made by ingestion program as input and compare to the solution file and produce
a learning curve.
Previous updates:
20190429: [ZY] Remove useless code block such as the function is_started;
               Better code layout.
20190426.4: [ZY] Fix yaml format in scores.txt (add missing spaces)
20190426.3: [ZY] Use f.write instead of yaml.dump to write scores.txt
20190426.2: [ZY] Add logging info when writing scores and learning curves.
20190426: [ZY] Now write to scores.txt whenever a new prediction is made. This
               way, participants can still get a score when they exceed time
               limit (but the submission's status will be marked as 'Failed').
20190425: [ZY] Add ScoringError and IngestionError: throw error in these cases.
               Participants will get 'Failed' for their error. But a score will
               still by computed if applicable.
               Improve importing order.
               Log CPU usage.
20190424: [ZY] Use logging instead of logger; remove start.txt checking.
20190424: [ZY] Add version and description.
20190419: [ZY] Judge if ingestion is alive by duration.txt; use logger."""

# Scoring program for the AutoDL challenge
# Isabelle Guyon and Zhengying Liu, ChaLearn, April 2018-

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.


################################################################################
# User defined constants
################################################################################

# Time budget for ingestion program (and thus for scoring)
# This is needed since scoring program is running all along with ingestion
# program in parallel. So we need to know how long ingestion program will run.
TIME_BUDGET = 7200

# Redirect stardant output to live results page (detailed_results.html)
# to have live output for debugging
REDIRECT_STDOUT = False

# Verbosity level of logging.
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
verbosity_level = 'INFO'

# Constant used for a missing score
missing_score = -0.999999

from functools import partial
from libscores import read_array, sp, ls, mvmean
from os import getcwd as pwd
from os.path import join
from sys import argv
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
import base64
import datetime
import logging
import matplotlib; matplotlib.use('Agg') # Solve the Tkinter display issue of matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import sys
import time
import yaml

# Set logging format to something like:
# 2019-04-25 12:52:51 INFO score.py: <message>
logging.basicConfig(
    level=getattr(logging, verbosity_level),
    format='%(asctime)s %(levelname)s %(filename)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

################################################################################
# Functions
################################################################################

def _HERE(*args):
    """Helper function for getting the current directory of the script."""
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(h, *args))

# Default I/O directories:
root_dir = _HERE(os.pardir)
default_solution_dir = join(root_dir, "AutoDL_sample_data")
default_prediction_dir = join(root_dir, "AutoDL_sample_result_submission")
default_score_dir = join(root_dir, "AutoDL_scoring_output")

# Metric used to compute the score of a point on the learning curve
def autodl_bac(solution, prediction):
  """Compute the normalized balanced accuracy.

  Args:
    solution: numpy.ndarray of shape (num_examples, num_classes), in this first
        edition of AutoDL challenge, all entries will be 0 or 1.
    prediction: numpy.ndarray of shape (num_examples, num_classes). Prediction
        made by Model.test(). All entries should be between 0 and 1.
  Returns:
    score: a float representing the normalized balanced accuracy, i.e. 2*BAC - 1
        where BAC is the balanced accuracy ( (TPR+TNR)/2 ) over all classes.
  """
  label_num = solution.shape[1]
  score = np.zeros(label_num)
  binarize=False
  if binarize:
    # Binarize prediction by thresholding at 0.5 (assumes predictions are like probabilities; we can also threshold at 0 if we want but we need to let the participants know)
    bin_prediction = np.zeros(prediction.shape)
    threshold = 0.5
    bin_prediction[prediction >= threshold] = 1
  else:
    # Participant's prediction is already binary (or at least in [0,1])
    bin_prediction = prediction
  # Compute the confusion matrix statistics
  tn = sum(np.multiply((1 - solution), (1 - bin_prediction)))
  fn = sum(np.multiply(solution, (1 - bin_prediction)))
  tp = sum(np.multiply(solution, bin_prediction))
  fp = sum(np.multiply((1 - solution), bin_prediction))
  # Bounding to avoid division by 0
  eps = 1e-15
  tp = sp.maximum(eps, tp)
  pos_num = sp.maximum(eps, tp + fn)
  tpr = tp / pos_num  # true positive rate (sensitivity)
  tn = sp.maximum(eps, tn)
  neg_num = sp.maximum(eps, tn + fp)
  tnr = tn / neg_num  # true negative rate (specificity)
  # Compute bac
  bac = 0.5 * (tpr + tnr)
  bac = mvmean(bac)  # average over all classes using moving average (better for numerical reasons)
  # Normalize: 0 for random, 1 for perfect
  score = 2*bac - 1
  return score

def is_one_hot_vector(x, axis=None, keepdims=False):
  """Check if a vector 'x' is one-hot (i.e. one entry is 1 and others 0)."""
  norm_1 = np.linalg.norm(x, ord=1, axis=axis, keepdims=keepdims)
  norm_inf = np.linalg.norm(x, ord=np.inf, axis=axis, keepdims=keepdims)
  return np.logical_and(norm_1 == 1, norm_inf == 1)

def is_multiclass(solution):
  """Return if a task is a multi-class classification task, i.e.  each example
  only has one label and thus each binary vector in `solution` only has
  one '1' and all the rest components are '0'.

  This function is useful when we want to compute metrics (e.g. accuracy) that
  are only applicable for multi-class task (and not for multi-label task).

  Args:
    solution: a numpy.ndarray object of shape [num_examples, num_classes].
  """
  return all(is_one_hot_vector(solution, axis=1))

def accuracy(solution, prediction):
  """Get accuracy of 'prediction' w.r.t true labels 'solution'."""
  epsilon = 1e-15
  # normalize prediction
  prediction_normalized =\
    prediction / (np.sum(np.abs(prediction), axis=1, keepdims=True) + epsilon)
  return np.sum(solution * prediction_normalized) / solution.shape[0]

def get_prediction_files(prediction_dir, basename, start):
  """Return prediction files for the task <basename>.

  Examples of prediction file name: mini.predict_0, mini.predict_1
  """
  prediction_files = ls(os.path.join(prediction_dir, basename + '*.predict_*'))
  # Exclude all files (if any) generated before start
  prediction_files = [f for f in prediction_files if os.path.getmtime(f)> start]
  return prediction_files

def get_fig_name(basename):
  """Helper function for getting learning curve figure name."""
  fig_name = "learning-curve-" + basename + ".png"
  return fig_name

def get_basename(solution_file):
  """
  Args:
    solution_file: string, e.g. '..../haha/munster.solution'
  Returns:
    basename: string, e.g. 'munster'
  """
  return solution_file.split(os.sep)[-1].split('.')[0]

# TODO: change this function to avoid repeated computing
def draw_learning_curve(solution_file, prediction_files,
                        scoring_function, output_dir,
                        basename, start, is_multiclass_task):
  """Draw learning curve for one task."""
  solution = read_array(solution_file) # numpy array
  scores = []
  roc_auc_scores = []
  timestamps = []
  if is_multiclass_task:
    accuracy_scores = []
  for prediction_file in prediction_files:
    timestamp = os.path.getmtime(prediction_file)
    prediction = read_array(prediction_file) # numpy array
    if (solution.shape != prediction.shape): raise ValueError(
        "Bad prediction shape: {}. ".format(prediction.shape) +
        "Expected shape: {}".format(solution.shape))
    scores.append(scoring_function(solution, prediction))
    try: # if only one class present in y_true. ROC AUC score is not defined in that case.
        roc_auc_scores.append(roc_auc_score(solution, prediction))
    except:
        roc_auc_scores.append(-1)
    timestamps.append(timestamp)
    if is_multiclass_task:
      accuracy_scores.append(accuracy(solution, prediction))
  # Sort two lists according to timestamps
  sorted_pairs = sorted(zip(timestamps, scores))
  roc_auc_sorted_pairs = sorted(zip(timestamps, roc_auc_scores))
  time_used = -1

  if len(timestamps) > 0:
    time_used = sorted_pairs[-1][0] - start
    latest_nbac = sorted_pairs[-1][1]
    latest_roc_auc = roc_auc_sorted_pairs[-1][1]
    logging.info("NBAC (2 * BAC - 1) of the latest prediction is {:.4f}."\
              .format(latest_nbac))
    if not latest_roc_auc == -1:
      logging.info("ROC AUC of the latest prediction is {:.4f}."\
                .format(latest_roc_auc))
    if is_multiclass_task:
      sorted_pairs_acc = sorted(zip(timestamps, accuracy_scores))
      latest_acc = sorted_pairs_acc[-1][1]
      logging.info("Accuracy of the latest prediction is {:.4f}."\
                .format(latest_acc))
  X = [t - start + 1 for t,_ in sorted_pairs] # Since X on log scale, set first x=1
  Y = [s for _,s in sorted_pairs]
  # Add origin as the first point of the curve
  X.insert(0, 1) # X starts from 1 to use log
  Y.insert(0, 0)
  # Truncate X using X_max
  X_max = TIME_BUDGET
  Y_max = 1
  log_X = [np.log(x+1)/np.log(X_max+1) for x in X if x <= X_max] # log_X \in [0, 1]
  log_X_max = 1
  X = X[:len(log_X)]
  Y = Y[:len(log_X)]
  # Draw learning curve
  plt.clf()
  fig, ax = plt.subplots(figsize=(7, 7.07)) #Have a small area of negative score
  ax.plot(X, Y, marker="o", label="Test score", markersize=3)
  # ax.step(X, Y, marker="o", label="Test score", markersize=3, where='post')
  # Add a point on the final line using last prediction
  X.append(TIME_BUDGET)
  Y.append(Y[-1])
  log_X.append(1)
  if len(log_X) >= 2:
    alc = area_under_learning_curve(log_X,Y)
  else:
    alc = 0
  ax.fill_between(X, Y, color='cyan')
  # ax.fill_between(X, Y, color='cyan', step='post')
  ax.text(X[-1], Y[-1], "{:.4f}".format(Y[-1])) # Show the latest/final score
  ax.plot(X[-2:], Y[-2:], '--') # Draw a dotted line from last prediction
  plt.title("Task: " + basename + " - Current normalized ALC: " + format(alc, '.4f'))
  plt.xlabel('time/second (log scale)')
  plt.xlim(left=1, right=X_max)
  plt.xscale('log')
  plt.ylabel('score (2*BAC - 1)')
  plt.ylim(bottom=-0.01, top=1)
  ax.grid(True, zorder=5)
  plt.legend()
  fig_name = get_fig_name(basename)
  path_to_fig = os.path.join(output_dir, fig_name)
  plt.savefig(path_to_fig)
  return alc, time_used

def area_under_learning_curve(X,Y):
  return auc(X,Y)

def init_scores_html(detailed_results_filepath):
  html_head = '<html><head> <meta http-equiv="refresh" content="5"> ' +\
              '</head><body><pre>'
  html_end = '</pre></body></html>'
  with open(detailed_results_filepath, 'a') as html_file:
    html_file.write(html_head)
    html_file.write("Starting training process... <br> Please be patient. " +
                    "Learning curves will be generated when first " +
                    "predictions are made.")
    html_file.write(html_end)

def write_scores_html(score_dir, auto_refresh=True, append=REDIRECT_STDOUT):
  filename = 'detailed_results.html'
  image_paths = sorted(ls(os.path.join(score_dir, '*.png')))
  if auto_refresh:
    html_head = '<html><head> <meta http-equiv="refresh" content="5"> ' +\
                '</head><body><pre>'
  else:
    html_head = """<html><body><pre>"""
  html_end = '</pre></body></html>'
  if append:
    mode = 'a'
  else:
    mode = 'w'
  filepath = os.path.join(score_dir, filename)
  with open(filepath, mode) as html_file:
      html_file.write(html_head)
      for image_path in image_paths:
        with open(image_path, "rb") as image_file:
          encoded_string = base64.b64encode(image_file.read())
          encoded_string = encoded_string.decode('utf-8')
          s = '<img src="data:image/png;charset=utf-8;base64,{}"/>'\
              .format(encoded_string)
          html_file.write(s + '<br>')
      html_file.write(html_end)
  logging.info("Wrote learning curve page to {}".format(filepath))

def write_score(score_dir, score, duration=-1):
  """Write score and duration to score_dir/scores.txt"""
  score_filename = os.path.join(score_dir, 'scores.txt')
  with open(score_filename, 'w') as f:
    f.write('score: ' + str(score) + '\n')
    f.write('Duration: ' + str(duration) + '\n')
  logging.debug("Wrote to score_filename={} with score={}, duration={}"\
                .format(score_filename, score, duration))

def list_files(startpath):
    """List a tree structure of directories and files from startpath"""
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        logging.debug('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            logging.debug('{}{}'.format(subindent, f))

def get_ingestion_pid(prediction_dir):
  """Get ingestion's process ID.
  """
  start_filepath = os.path.join(prediction_dir, 'start.txt')
  with open(start_filepath, 'r') as f:
    pid = int(f.readline().split(':')[-1])
  return pid

def ingestion_is_alive(prediction_dir):
  """Check if ingestion is still alive by checking if the file 'duration.txt'
  if generated in the folder of predictions.
  """
  duration_filepath =  os.path.join(prediction_dir, 'duration.txt')
  logging.debug("CPU usage: {}%".format(psutil.cpu_percent()))
  logging.debug("Virtual memory: {}".format(psutil.virtual_memory()))
  return not os.path.isfile(duration_filepath)

def is_process_alive(pid):
  try:
    os.kill(ingestion_pid, 0)
  except OSError:
    return False
  else:
    return True

class IngestionError(Exception):
  pass

class ScoringError(Exception):
  pass


# =============================== MAIN ========================================

if __name__ == "__main__":

    the_date = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 1:  # Use the default data directories if no arguments are provided
        solution_dir = default_solution_dir
        prediction_dir = default_prediction_dir
        score_dir = default_score_dir
     # the case for indicating special input data dir
     # this is used especially in `test_with_baseline.py`
    elif len(argv) == 2:
        solution_dir = argv[1]
        prediction_dir = default_prediction_dir
        score_dir = default_score_dir
    elif len(argv) == 3: # The current default configuration of Codalab
        solution_dir = os.path.join(argv[1], 'ref')
        prediction_dir = os.path.join(argv[1], 'res')
        score_dir = argv[2]
    elif len(argv) == 4:
        solution_dir = argv[1]
        prediction_dir = os.path.join(argv[2], 'res')
        score_dir = argv[3]
    else:
        swrite('\n*** WRONG NUMBER OF ARGUMENTS ***\n\n')
        exit(1)

    # Create the output directory, if it does not already exist and open output files
    if not os.path.isdir(score_dir):
      os.mkdir(score_dir)
    detailed_results_filepath = os.path.join(score_dir, 'detailed_results.html')
    # Initialize detailed_results.html
    init_scores_html(detailed_results_filepath)

    # Redirect standard output to detailed_results.html to have real-time
    # feedback for debugging
    if REDIRECT_STDOUT:
      if not os.path.exists(score_dir):
        os.makedirs(score_dir)
      detailed_results_filepath = os.path.join(score_dir,
                                               'detailed_results.html')
      logging.basicConfig(filename=detailed_results_filepath)
      logging.info("""<html><head> <meta http-equiv="refresh" content="5"> </head><body><pre>""")
      logging.info("Redirecting standard output. " +
                "Please check out output at {}."\
                .format(detailed_results_filepath))

    logging.info("Version: {}. Description: {}".format(VERSION, DESCRIPTION))

    logging.debug("sys.argv = " + str(sys.argv))
    with open(os.path.join(os.path.dirname(sys.argv[0]), 'metadata'), 'r') as f:
      logging.debug("Content of the metadata file: ")
      logging.debug(str(f.read()))
    logging.debug("Using solution_dir: " + str(solution_dir))
    logging.debug("Using prediction_dir: " + str(prediction_dir))
    logging.debug("Using score_dir: " + str(score_dir))
    logging.debug("Scoring datetime: " + str(the_date))

    # Use the timestamp of 'detailed_results.html' as start time
    # This is more robust than using start = time.time()
    # especially when Docker image time is not synced with host time
    start = os.path.getmtime(detailed_results_filepath)
    start_str = time.ctime(start)
    logging.info("Start scoring program at " + start_str)

    # Get the metric
    scoring_function = autodl_bac
    metric_name = "Area under Learning Curve"

    # Get all the solution files from the solution directory
    solution_names = sorted(ls(os.path.join(solution_dir, '*.solution')))
    if len(solution_names) > 1: # Assert only one file is found
      raise ValueError("Multiple solution files found: {}!"\
                       .format(solution_names))
    solution_file = solution_names[0]
    solution = read_array(solution_file)
    is_multiclass_task = is_multiclass(solution)
    # Extract the dataset name from the file name
    basename = get_basename(solution_file)
    nb_preds = {x:0 for x in solution_names}
    scores = {x:missing_score for x in solution_names}

    scoring_success = True

    try:
      # Begin scoring process, along with ingestion program
      # Moniter training processes while time budget is not attained
      while(time.time() < start + TIME_BUDGET):
        time.sleep(0.5)
        # Give list of prediction files
        prediction_files = get_prediction_files(prediction_dir, basename, start)
        nb_preds_old = nb_preds[solution_file]
        nb_preds_new = len(prediction_files)
        if(nb_preds_new > nb_preds_old):
          now = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
          logging.info("[+] New prediction found. Now number of predictions " +
                       "made = " + str(nb_preds_new))
          alc = 0
          alc, time_used = draw_learning_curve(solution_file=solution_file,
                                    prediction_files=prediction_files,
                                    scoring_function=scoring_function,
                                    output_dir=score_dir,
                                    basename=basename,
                                    start=start,
                                    is_multiclass_task=is_multiclass_task)
          nb_preds[solution_file] = nb_preds_new
          scores[solution_file] = alc
          logging.info("Current area under learning curve for {}: {:.4f}"\
                    .format(basename, scores[solution_file]))
          # Update learning curve page (detailed_results.html)
          write_scores_html(score_dir)
          # Write score
          write_score(score_dir, float(alc), duration=time_used)

        if not ingestion_is_alive(prediction_dir):
          logging.info("Detected ingestion program is not running. " +
                       "Stop scoring now.")
          break

    except Exception as e:
      scoring_success = False
      logging.error("[-] Error occurred in scoring:\n" + str(e),
                    exc_info=True)

    # Write one last time the detailed results page without auto-refreshing
    write_scores_html(score_dir, auto_refresh=False)
    # Write score
    score = scores[solution_file]

    # Read the execution time and add it to scores.txt
    # Spend 30 seconds to search for a duration.txt file
    # Use 'duration.txt' file to detect if ingestion program exits early
    duration_filepath =  os.path.join(prediction_dir, 'duration.txt')
    duration = None
    if not os.path.isfile(duration_filepath) or not scoring_success:
      logging.error("[-] Some error occurred in scoring program. " +
                  "Please see output/error log of Scoring Step.")
      raise ScoringError("Scoring Step terminated abnormally. " +
                       "Please see output/error log of Scoring Step.")
    else:
      with open(duration_filepath, 'r') as f:
        duration_dict = yaml.safe_load(f)
      duration = duration_dict['Duration']
      write_score(score_dir, score, duration=duration)

      if duration_dict['Success'] == 0:
        logging.error("[-] Some error occurred in ingestion program. " +
                    "Please see output/error log of Ingestion Step.")
        raise IngestionError("Ingestion Step terminated abnormally. " +
                         "Please see output log of Ingestion Step.")
      else:
        logging.info("[+] Successfully finished scoring! " +\
                  "Duration used: {:.2f} sec. ".format(duration) +\
                  "The score of your algorithm on this task ({}) is: {:.6f}."\
                  .format(basename, score))

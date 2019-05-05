################################################################################
# Name:         Scoring Program
# Author:       Zhengying Liu, Isabelle Guyon, Adrien Pavao, Zhen Xu
# Update time:  5 May 2019
# Usage: 		python score.py --solution_dir=<solution_dir> --prediction_dir=<prediction_dir> --score_dir=<score_dir>
#           solution_dir contains  e.g. adult.solution
#           prediction_dir should contain e.g. start.txt, adult.predict_0, adult.predict_1,..., end.txt.
#           score_dir should contain scores.txt, detailed_results.html

VERSION = 'v20190505'
DESCRIPTION =\
"""This is the scoring program for AutoDL challenge. It takes the predictions
made by ingestion program as input and compare to the solution file and produce
a learning curve.
Previous updates:
20190505: [ZY] Use argparse to parse directories AND time budget;
               Fix nb_preds not updated error.
20190504: [ZY] Don't raise Exception anymore (for most cases) in order to
               always have 'Finished' for each submission;
               Kill ingestion when time limit is exceeded;
               Use the last modified time of the file 'start.txt' written by
               ingestion as the start time (`ingestion_start`);
               Use module-specific logger instead of logging (with root logger);
               Use the function update_score_and_learning_curve;
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


################################################################################
# User defined constants
################################################################################

# Verbosity level of logging.
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
verbosity_level = 'INFO'

# Redirect stardant output to live results page (detailed_results.html)
# to have live output for debugging
REDIRECT_STDOUT = False

# Constant used for a missing score
missing_score = -0.999999

from functools import partial
from libscores import read_array, sp, ls, mvmean
from os import getcwd as pwd
from os.path import join
from sys import argv
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
import argparse
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

def get_logger(verbosity_level, use_error_log=False):
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
  logger.addHandler(stdout_handler)
  if use_error_log:
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
  logger.propagate = False
  return logger

logger = get_logger(verbosity_level)

################################################################################
# Functions
################################################################################

def _HERE(*args):
    """Helper function for getting the current directory of the script."""
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(h, *args))

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
  prediction_files = [f for f in prediction_files
                      if os.path.getmtime(f) >= start]
  excluded_files = {f:os.path.getmtime(f) for f in prediction_files
                      if os.path.getmtime(f) < start}
  if excluded_files:
    logger.debug("Some predictions are made before ingestion start time: {}"\
                 .format(start))
    logger.debug("These files are: {}".format(excluded_files))
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
    logger.info("NBAC (2 * BAC - 1) of the latest prediction is {:.4f}."\
              .format(latest_nbac))
    if not latest_roc_auc == -1:
      logger.info("ROC AUC of the latest prediction is {:.4f}."\
                .format(latest_roc_auc))
    if is_multiclass_task:
      sorted_pairs_acc = sorted(zip(timestamps, accuracy_scores))
      latest_acc = sorted_pairs_acc[-1][1]
      logger.info("Accuracy of the latest prediction is {:.4f}."\
                .format(latest_acc))
  X = [t - start + 1 for t,_ in sorted_pairs] # Since X on log scale, set first x=1
  Y = [s for _,s in sorted_pairs]
  # Add origin as the first point of the curve
  X.insert(0, 1) # X starts from 1 to use log
  Y.insert(0, 0)
  # Truncate X using X_max
  X_max = time_budget
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
  X.append(time_budget)
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
  plt.close()
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
  logger.debug("Wrote learning curve page to {}".format(filepath))

def write_score(score_dir, score, duration=-1):
  """Write score and duration to score_dir/scores.txt"""
  score_filename = os.path.join(score_dir, 'scores.txt')
  with open(score_filename, 'w') as f:
    f.write('score: ' + str(score) + '\n')
    f.write('Duration: ' + str(duration) + '\n')
  logger.debug("Wrote to score_filename={} with score={}, duration={}"\
                .format(score_filename, score, duration))

def update_score_and_learning_curve(prediction_dir,
                                    basename,
                                    start,
                                    solution_file,
                                    scoring_function,
                                    score_dir,
                                    is_multiclass_task):
  prediction_files = get_prediction_files(prediction_dir, basename, start)
  alc = 0
  alc, time_used = draw_learning_curve(solution_file=solution_file,
                            prediction_files=prediction_files,
                            scoring_function=scoring_function,
                            output_dir=score_dir,
                            basename=basename,
                            start=start,
                            is_multiclass_task=is_multiclass_task)
  # Update learning curve page (detailed_results.html)
  write_scores_html(score_dir)
  # Write score
  score = float(alc)
  write_score(score_dir, score, duration=time_used)
  return score

def list_files(startpath):
    """List a tree structure of directories and files from startpath"""
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        logger.debug('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            logger.debug('{}{}'.format(subindent, f))

def get_ingestion_info(prediction_dir):
  """Get info on ingestion program: PID, start time, etc.

  Args:
    prediction_dir: a string, directory containing predictions (output of
      ingestion)
  Returns:
    A dictionary with keys 'ingestion_pid' and 'start_time' if the file
      'start.txt' exists. Otherwise return `None`.
  """
  start_filepath = os.path.join(prediction_dir, 'start.txt')
  if os.path.exists(start_filepath):
    with open(start_filepath, 'r') as f:
      ingestion_info = yaml.safe_load(f)
    return ingestion_info
  else:
    return None

def get_ingestion_start_time(prediction_dir):
  """
  Returns:
    a float, the last modification time of the file 'start.txt', written by
      ingestion at its beginning.
  """
  start_filepath = os.path.join(prediction_dir, 'start.txt')
  if os.path.exists(start_filepath):
    last_modified_time = os.path.getmtime(start_filepath)
    with open(start_filepath, 'r') as f:
      ingestion_info = yaml.safe_load(f)
    start_time_written_by_ingestion = ingestion_info['start_time']
    if np.abs(start_time_written_by_ingestion - last_modified_time) > 5:
      logger.error("Considerable difference between time got by " +
                   "time.time(): {} ".format(start_time_written_by_ingestion) +
                   "(the value of start_time in 'start.txt') "
                   "and os.path.getmtime('start.txt'): {}. "\
                   .format(last_modified_time) +
                   "This might be due to the inconsistency of the File " +
                   "System time and the system time."
                   "Using the latter as ingestion start time.")
    return last_modified_time
  else:
    return None

def ingestion_is_alive(prediction_dir):
  """Check if ingestion is still alive by checking if the file 'end.txt'
  is generated in the folder of predictions.
  """
  end_filepath =  os.path.join(prediction_dir, 'end.txt')
  logger.debug("CPU usage: {}%".format(psutil.cpu_percent()))
  logger.debug("Virtual memory: {}".format(psutil.virtual_memory()))
  return not os.path.isfile(end_filepath)

def is_process_alive(pid):
  try:
    os.kill(ingestion_pid, 0)
  except OSError:
    return False
  else:
    return True

def terminate_process(pid):
  process = psutil.Process(ingestion_pid)
  process.terminate()
  logger.debug("Terminated process with pid={} in scoring.".format(pid))

class IngestionError(Exception):
  pass

class ScoringError(Exception):
  pass


# =============================== MAIN ========================================

if __name__ == "__main__":

    scoring_start = time.time()

    # Default I/O directories:
    root_dir = _HERE(os.pardir)
    default_solution_dir = join(root_dir, "AutoDL_sample_data")
    default_prediction_dir = join(root_dir, "AutoDL_sample_result_submission")
    default_score_dir = join(root_dir, "AutoDL_scoring_output")
    default_time_budget = 7200

    # Parse directories from input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--solution_dir', type=str,
                        default=default_solution_dir,
                        help="Directory storing the solution with true " +
                             "labels, e.g. adult.solution.")
    parser.add_argument('--prediction_dir', type=str,
                        default=default_prediction_dir,
                        help="Directory storing the predictions. It should" +
                             "contain e.g. [start.txt, adult.predict_0, " +
                             "adult.predict_1, ..., end.txt].")
    parser.add_argument('--score_dir', type=str,
                        default=default_score_dir,
                        help="Directory storing the scoring output " +
                             "e.g. `scores.txt` and `detailed_results.html`.")
    parser.add_argument('--time_budget', type=float,
                        default=default_time_budget,
                        help="Time budget for running ingestion program.")
    args = parser.parse_args()
    logger.debug("Parsed args are: " + str(args))
    logger.debug("-" * 50)
    solution_dir = args.solution_dir
    prediction_dir = args.prediction_dir
    score_dir = args.score_dir
    time_budget = args.time_budget

    # Create the output directory, if it does not already exist and open output files
    if not os.path.isdir(score_dir):
      os.mkdir(score_dir)
    detailed_results_filepath = os.path.join(score_dir, 'detailed_results.html')
    # Initialize detailed_results.html
    init_scores_html(detailed_results_filepath)
    # Write initial score to `missing_score`
    write_score(score_dir, missing_score, duration=0)

    # Redirect standard output to detailed_results.html to have real-time
    # feedback for debugging
    if REDIRECT_STDOUT:
      if not os.path.exists(score_dir):
        os.makedirs(score_dir)
      detailed_results_filepath = os.path.join(score_dir,
                                               'detailed_results.html')
      logging.basicConfig(filename=detailed_results_filepath)
      logger.info("""<html><head> <meta http-equiv="refresh" content="5"> </head><body><pre>""")
      logger.info("Redirecting standard output. " +
                "Please check out output at {}."\
                .format(detailed_results_filepath))

    logger.debug("Version: {}. Description: {}".format(VERSION, DESCRIPTION))

    logger.debug("sys.argv = " + str(sys.argv))
    with open(os.path.join(os.path.dirname(sys.argv[0]), 'metadata'), 'r') as f:
      logger.debug("Content of the metadata file: ")
      logger.debug(str(f.read()))
    logger.debug("Using solution_dir: " + str(solution_dir))
    logger.debug("Using prediction_dir: " + str(prediction_dir))
    logger.debug("Using score_dir: " + str(score_dir))

    # Wait 30 seconds for ingestion to start and write 'start.txt',
    # Otherwise, raise an exception.
    wait_time = 30
    ingestion_info = None
    for i in range(wait_time):
      ingestion_info = get_ingestion_info(prediction_dir)
      if not ingestion_info is None:
        logger.info("Detected the start of ingestion after {} ".format(i) +
                    "seconds. Start scoring.")
        break
      time.sleep(1)
    else:
      raise IngestionError("[-] Failed: scoring didn't detected the start of " +
                           "ingestion after {} seconds.".format(wait_time))

    # Get ingestion start time
    ingestion_start = get_ingestion_start_time(prediction_dir)
    logger.debug("Ingestion start time: {}".format(ingestion_start))
    logger.debug("Scoring start time: {}".format(scoring_start))

    # Get ingestion PID
    ingestion_pid = ingestion_info['ingestion_pid']

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
    time_limit_exceeded = False

    try:
      # Begin scoring process, along with ingestion program
      # Moniter training processes while time budget is not attained
      while(time.time() < ingestion_start + time_budget):
        time.sleep(0.5)
        # Give list of prediction files
        prediction_files = get_prediction_files(prediction_dir, basename,
                                                ingestion_start)
        nb_preds_old = nb_preds[solution_file]
        nb_preds_new = len(prediction_files)
        if(nb_preds_new > nb_preds_old):
          logger.info("[+] New prediction found. Now number of predictions " +
                       "made = " + str(nb_preds_new))
          score = update_score_and_learning_curve(prediction_dir,
                                                  basename,
                                                  ingestion_start,
                                                  solution_file,
                                                  scoring_function,
                                                  score_dir,
                                                  is_multiclass_task)
          nb_preds[solution_file] = nb_preds_new
          logger.info("Current area under learning curve for {}: {:.4f}"\
                    .format(basename, score))

        if not ingestion_is_alive(prediction_dir):
          logger.info("Detected ingestion program had stopped running " +
                      "because an 'end.txt' file is written by ingestion."
                      "Stop scoring now.")
          break
      else: # When time budget is used up, kill ingestion
        time_limit_exceeded = True
        terminate_process(ingestion_pid)
        logger.info("Detected time budget is used up. Killed ingestion and " +
                    "terminating scoring...")
    except Exception as e:
      scoring_success = False
      logger.error("[-] Error occurred in scoring:\n" + str(e),
                    exc_info=True)

    score = update_score_and_learning_curve(prediction_dir,
                                            basename,
                                            ingestion_start,
                                            solution_file,
                                            scoring_function,
                                            score_dir,
                                            is_multiclass_task)
    logger.info("Final area under learning curve for {}: {:.4f}"\
              .format(basename, score))

    # Write one last time the detailed results page without auto-refreshing
    write_scores_html(score_dir, auto_refresh=False)

    # Use 'end.txt' file to detect if ingestion program ends
    end_filepath =  os.path.join(prediction_dir, 'end.txt')
    if not scoring_success:
      logger.error("[-] Some error occurred in scoring program. " +
                  "Please see output/error log of Scoring Step.")
    elif not os.path.isfile(end_filepath):
      if time_limit_exceeded:
        logger.error("[-] Ingestion program exceeded time budget. " +
                     "Predictions made so far will be used for evaluation.")
      else: # Less probable to fall in this case
        if is_process_alive(ingestion_pid):
          terminate_process(ingestion_pid)
        logger.error("[-] No 'end.txt' file is produced by ingestion. " +
                     "Ingestion or scoring may have not terminated normally.")
    else:
      with open(end_filepath, 'r') as f:
        end_info_dict = yaml.safe_load(f)
      ingestion_duration = end_info_dict['ingestion_duration']

      if end_info_dict['ingestion_success'] == 0:
        logger.error("[-] Some error occurred in ingestion program. " +
                    "Please see output/error log of Ingestion Step.")
      else:
        logger.info("[+] Successfully finished scoring! " +
                  "Scoring duration: {:.2f} sec. "\
                  .format(time.time() - scoring_start) +
                  "Ingestion duration: {:.2f} sec. "\
                  .format(ingestion_duration) +
                  "The score of your algorithm on the task '{}' is: {:.6f}."\
                  .format(basename, score))

    logger.info("[Scoring terminated]")

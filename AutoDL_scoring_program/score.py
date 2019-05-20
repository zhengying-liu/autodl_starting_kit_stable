################################################################################
# Name:         Scoring Program
# Author:       Zhengying Liu, Isabelle Guyon, Adrien Pavao, Zhen Xu
# Update time:  5 May 2019
# Usage: 		python score.py --solution_dir=<solution_dir> --prediction_dir=<prediction_dir> --score_dir=<score_dir>
#           solution_dir contains  e.g. adult.solution
#           prediction_dir should contain e.g. start.txt, adult.predict_0, adult.predict_1,..., end.txt.
#           score_dir should contain scores.txt, detailed_results.html

VERSION = 'v20190519'
DESCRIPTION =\
"""This is the scoring program for AutoDL challenge. It takes the predictions
made by ingestion program as input and compare to the solution file and produce
a learning curve.
Previous updates:
20190519: [ZY] Use the correct function for computing AUC of step functions
20190516: [ZY] Change time budget to 20 minutes.
20190508: [ZY] Decompose drawing learning curve functions;
               Remove redirect output feature;
               Use AUC instead of BAC;
               Ignore columns with only one class when computing AUC;
               Use step function instead of trapezoidal rule;
               Change t0=300 from t0=1 in time transformation:
                 log(1 + t/t0) / log(1 + T/t0)
               Add second x-axis for the real time in seconds
20190505: [ZY] Use argparse to parse directories AND time budget;
               Fix num_preds not updated error.
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

# Constant used for a missing score
missing_score = -0.999999

from functools import partial
from libscores import read_array, sp, ls, mvmean, tiedrank
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
def autodl_auc(solution, prediction, valid_columns_only=True):
  """Compute normarlized Area under ROC curve (AUC).
  Return Gini index = 2*AUC-1 for  binary classification problems.
  Should work for a vector of binary 0/1 (or -1/1)"solution" and any discriminant values
  for the predictions. If solution and prediction are not vectors, the AUC
  of the columns of the matrices are computed and averaged (with no weight).
  The same for all classification problems (in fact it treats well only the
  binary and multilabel classification problems). When `valid_columns` is not
  `None`, only use a subset of columns for computing the score.
  """
  if valid_columns_only:
    valid_columns = get_valid_columns(solution)
    if len(valid_columns) < solution.shape[-1]:
      logger.warning("Some columns in solution have only one class, " +
                     "ignoring these columns for evaluation.")
    solution = solution[:, valid_columns].copy()
    prediction = prediction[:, valid_columns].copy()
  label_num = solution.shape[1]
  auc = np.empty(label_num)
  for k in range(label_num):
    r_ = tiedrank(prediction[:, k])
    s_ = solution[:, k]
    if sum(s_) == 0: print("WARNING: no positive class example in class {}"\
                           .format(k + 1))
    npos = sum(s_ == 1)
    nneg = sum(s_ < 1)
    auc[k] = (sum(r_[s_ == 1]) - npos * (npos + 1) / 2) / (nneg * npos)
  return 2 * mvmean(auc) - 1

def accuracy(solution, prediction):
  """Get accuracy of 'prediction' w.r.t true labels 'solution'."""
  epsilon = 1e-15
  # normalize prediction
  prediction_normalized =\
    prediction / (np.sum(np.abs(prediction), axis=1, keepdims=True) + epsilon)
  return np.sum(solution * prediction_normalized) / solution.shape[0]

def get_valid_columns(solution):
  """Get a list of column indices for which the column has more than one class.
  This is necessary when computing BAC or AUC which involves true positive and
  true negative in the denominator. When some class is missing, these scores
  don't make sense (or you have to add an epsilon to remedy the situation).

  Args:
    solution: array, a matrix of binary entries, of shape
      (num_examples, num_features)
  Returns:
    valid_columns: a list of indices for which the column has more than one
      class.
  """
  num_examples = solution.shape[0]
  col_sum = np.sum(solution, axis=0)
  valid_columns = np.where(1 - np.isclose(col_sum, 0) -
                               np.isclose(col_sum, num_examples))[0]
  return valid_columns

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

def get_fig_name(task_name):
  """Helper function for getting learning curve figure name."""
  fig_name = "learning-curve-" + task_name + ".png"
  return fig_name

def get_solution(solution_dir):
  """Get the solution array from solution directory."""
  solution_names = sorted(ls(os.path.join(solution_dir, '*.solution')))
  if len(solution_names) != 1: # Assert only one file is found
    logger.warning("{} solution files found: {}! "\
                   .format(len(solution_names), solution_names) +
                   "Return `None` as solution.")
    return None
  solution_file = solution_names[0]
  solution = read_array(solution_file)
  return solution

def get_task_name(solution_dir):
  """Get the task name from solution directory."""
  solution_names = sorted(ls(os.path.join(solution_dir, '*.solution')))
  if len(solution_names) != 1: # Assert only one file is found
    logger.warning("{} solution files found: {}! "\
                   .format(len(solution_names), solution_names) +
                   "Return `None` as task name.")
    return None
  solution_file = solution_names[0]
  task_name = solution_file.split(os.sep)[-1].split('.')[0]
  return task_name

def get_prediction_files(prediction_dir):
  """Return prediction files in prediction directory.

  Examples of prediction file name: mini.predict_0, mini.predict_1
  """
  prediction_files = ls(os.path.join(prediction_dir, '*.predict_*'))
  task_names = set([os.path.basename(f.split('.')[-2])
                    for f in prediction_files])
  if len(task_names) > 1:
    raise ValueError("Predictions of multiple tasks are found: {}!"\
                     .format(prediction_files))
  order_key = lambda filename: int(filename.split('_')[-1])
  prediction_files = sorted(prediction_files, key=order_key)
  return prediction_files

def get_new_prediction_files(prediction_dir, prediction_files_so_far=None):
  """Get a list of new predictions (arrays) w.r.t. `prediction_files_so_far`."""
  prediction_files = get_prediction_files(prediction_dir)
  if prediction_files_so_far is None:
    prediction_files_so_far = []
  new_prediction_files = [p for p in prediction_files
                           if p not in prediction_files_so_far]
  order_key = lambda filename: int(filename.split('_')[-1])
  new_prediction_files = sorted(new_prediction_files, key=order_key)
  # new_predictions = [read_array(p) for p in new_prediction_files]
  return new_prediction_files

def transform_time(t, T, t0=None):
  if t0 is None:
    t0 = T
  return np.log(1 + t / t0) / np.log(1 + T / t0)

def auc_step(X, Y):
  """Compute area under curve using step function (in 'post' mode)."""
  if len(X) != len(Y):
    raise ValueError("The length of X and Y should be equal but got " +
                     "{} and {} !".format(len(X), len(Y)))
  area = 0
  for i in range(len(X) - 1):
    delta_X = X[i + 1] - X[i]
    area += delta_X * Y[i]
  return area

def plot_learning_curve(timestamps, scores,
                        start_time=0, time_budget=7200, method='step',
                        transform=None, task_name=None, curve_color=None,
                        area_color='cyan', fill_area=True, model_name='',
                        clear_figure=True):
  """Plot learning curve using scores and corresponding timestamps.

  Args:
    timestamps: iterable of float, each element is the timestamp of
      corresponding performance. These timestamps should be INCREASING.
    scores: iterable of float, scores at each timestamp
    start_time: float, the start time, should be smaller than any timestamp
    time_budget: float, the time budget, should be larger than any timestamp
    method: string, can be one of ['step', 'trapez']
    transform: callable that transform [0, time_budget] into [0, 1]. If `None`,
      use the default transformation
          lambda t: np.log2(1 + t / time_budget)
    task_name: string, name of the task
    curve_color: matplotlib color, color of the learning curve
    area_color: matplotlib color, color of the area under learning curve
    fill_area: boolean, fill the area under the curve or not
    model_name: string, name of the model (learning algorithm).
    clear_figure: boolean, clear previous figures or not
  Returns:
    alc: float, the area under learning curve.
    ax: matplotlib.axes.Axes, the figure with learning curve
  Raises:
    ValueError: if the length of `timestamps` and `scores` are not equal,
      or if `timestamps` is not increasing, or if certain timestamp is not in
      the interval [start_time, start_time + time_budget], or if `method` has
      bad values.
  """
  le = len(timestamps)
  if not le == len(scores):
    raise ValueError("The number of timestamps {} ".format(le) +
                     "should be equal to the number of " +
                     "scores {}!".format(len(scores)))
  for i in range(le):
    if i < le - 1 and not timestamps[i] <= timestamps[i + 1]:
      raise ValueError("The timestamps should be increasing! But got " +
                       "[{}, {}] ".format(timestamps[i], timestamps[i + 1]) +
                       "at index [{}, {}].".format(i, i + 1))
    if timestamps[i] < start_time:
      raise ValueError("The timestamp {} at index {}".format(timestamps[i], i) +
                       " is earlier than start time {}!".format(start_time))
  timestamps = [t for t in timestamps if t <= time_budget + start_time]
  if len(timestamps) < le:
    logger.warning("Some predictions are made after the time budget! " +
                   "Ignoring all predictions from the index {}."\
                   .format(len(timestamps)))
    scores = scores[:len(timestamps)]
  if transform is None:
    t0 = 60
    # default transformation
    transform = lambda t: transform_time(t, time_budget, t0=t0)
    xlabel = "Transformed time: " +\
             r'$\tilde{t} = \frac{\log (1 + t / t_0)}{ \log (1 + T / t_0)}$ ' +\
             ' ($T = ' + str(int(time_budget)) + '$, ' +\
             ' $t_0 = ' + str(int(t0)) + '$)'
  else:
    xlabel = "Transformed time: " + r'$\tilde{t}$'
  relative_timestamps = [t - start_time for t in timestamps]
  # Transform X
  X = [transform(t) for t in relative_timestamps]
  Y = scores.copy()
  # Add origin as the first point of the curve
  X.insert(0, 0)
  Y.insert(0, 0)
  # Draw learning curve
  if clear_figure:
    plt.clf()
  fig, ax = plt.subplots(figsize=(7, 7.07)) #Have a small area of negative score
  if method == 'step':
    drawstyle = 'steps-post'
    step = 'post'
    auc_func = auc_step
  elif method == 'trapez':
    drawstyle = 'default'
    step = None
    auc_func = auc
  else:
    raise ValueError("The `method` variable should be one of " +
                     "['step', 'trapez']!")
  # Add a point on the final line using last prediction
  X.append(1)
  Y.append(Y[-1])
  # Compute AUC using step function rule or trapezoidal rule
  alc = auc_func(X, Y)
  # Plot the major part of the figure: the curve
  ax.plot(X[:-1], Y[:-1], drawstyle=drawstyle, marker="o",
          label=model_name + " ALC={:.4f}".format(alc),
          markersize=3, color=curve_color)
  # ax.set_label(model_name + " ALC={:.4f}".format(alc))
  # Fill area under the curve
  if fill_area:
    ax.fill_between(X, Y, color='cyan', step=step)
  # Show the latest/final score
  ax.text(X[-1], Y[-1], "{:.4f}".format(Y[-1]))
  # Draw a dotted line from last prediction
  ax.plot(X[-2:], Y[-2:], '--')
  plt.title("Learning curve for task: {}".format(task_name), y=1.06)
  ax.set_xlabel(xlabel)
  ax.set_xlim(left=0, right=1)
  ax.set_ylabel('score (2 * AUC - 1)')
  ax.set_ylim(bottom=-0.01, top=1)
  ax.grid(True, zorder=5)
  # Show real time in seconds in a second x-axis
  ax2 = ax.twiny()
  ticks = [10, 60, 300, 600, 1200] +\
          list(range(1800, int(time_budget) + 1, 1800))
  ax2.set_xticks([transform(t) for t in ticks])
  ax2.set_xticklabels(ticks)
  ax.legend()
  return alc, ax

# TODO: change this function to avoid repeated computing
def draw_learning_curve(solution_dir, prediction_files,
                        scoring_function, output_dir,
                        basename, start, is_multiclass_task):
  """Draw learning curve for one task."""
  solution = get_solution(solution_dir) # numpy array
  scores = []
  roc_auc_scores = []
  _, timestamps = get_timestamps(prediction_dir)
  if is_multiclass_task:
    accuracy_scores = []
  for prediction_file in prediction_files:
    prediction = read_array(prediction_file) # numpy array
    if (solution.shape != prediction.shape): raise ValueError(
        "Bad prediction shape: {}. ".format(prediction.shape) +
        "Expected shape: {}".format(solution.shape))
    scores.append(scoring_function(solution, prediction))
    try: # if only one class present in y_true. ROC AUC score is not defined in that case.
        roc_auc_scores.append(roc_auc_score(solution, prediction))
    except:
        roc_auc_scores.append(-1)
    if is_multiclass_task:
      accuracy_scores.append(accuracy(solution, prediction))
  # Sort two lists according to timestamps
  sorted_pairs = sorted(zip(timestamps, scores))
  roc_auc_sorted_pairs = sorted(zip(timestamps, roc_auc_scores))
  time_used = -1
  if len(timestamps) > 0:
    time_used = sorted_pairs[-1][0] - start
    latest_score = sorted_pairs[-1][1]
    latest_roc_auc = roc_auc_sorted_pairs[-1][1]
    logger.info("(2 * AUC - 1) of the latest prediction is {:.4f}."\
              .format(latest_score))
    if not latest_roc_auc == -1:
      logger.info("ROC AUC of the latest prediction is {:.4f}."\
                .format(latest_roc_auc))
    if is_multiclass_task:
      sorted_pairs_acc = sorted(zip(timestamps, accuracy_scores))
      latest_acc = sorted_pairs_acc[-1][1]
      logger.info("Accuracy of the latest prediction is {:.4f}."\
                .format(latest_acc))
  X = [t for t, _ in sorted_pairs]
  Y = [s for _, s in sorted_pairs]
  alc, ax = plot_learning_curve(X, Y,
                          start_time=start, time_budget=time_budget,
                          task_name=basename)
  fig_name = get_fig_name(basename)
  path_to_fig = os.path.join(output_dir, fig_name)
  plt.savefig(path_to_fig)
  plt.close()
  return alc, time_used

def update_score_and_learning_curve(prediction_dir,
                                    basename,
                                    start,
                                    solution_dir,
                                    scoring_function,
                                    score_dir,
                                    is_multiclass_task):
  prediction_files = get_prediction_files(prediction_dir)
  alc = 0
  alc, time_used = draw_learning_curve(solution_dir=solution_dir,
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

def write_scores_html(score_dir, auto_refresh=True, append=False):
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

def get_ingestion_info(prediction_dir):
  """Get info on ingestion program: PID, start time, etc. from 'start.txt'.

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

def get_timestamps(prediction_dir):
  """Read predictions' timestamps stored in 'start.txt'.

  The 'start.txt' file should be similar to
    ingestion_pid: 31315
    start_time: 1557269921.7939095
    0: 1557269953.5586617
    1: 1557269956.012751
    2: 1557269958.3
  We see there are 3 predictions. Then this function will return
    start_time, timestamps =
      1557269921.7939095, [1557269953.5586617, 1557269956.012751, 1557269958.3]
  """
  start_filepath = os.path.join(prediction_dir, 'start.txt')
  if os.path.exists(start_filepath):
    with open(start_filepath, 'r') as f:
      ingestion_info = yaml.safe_load(f)
    start_time = ingestion_info['start_time']
    timestamps = []
    idx = 0
    while idx in ingestion_info:
      timestamps.append(ingestion_info[idx])
      idx += 1
    return start_time, timestamps
  else:
    logger.warning("No 'start.txt' file found in the prediction directory " +
                   "{}. Return `None` as timestamps.")
    return None

def get_scores(scoring_function, solution, predictions):
  """Compute a list of scores for a list of predictions.

  Args:
    scoring_function: callable with signature
      scoring_function(solution, predictions)
    solution: Numpy array, the solution (true labels).
    predictions: list of array, predictions.
  Returns:
    a list of float, scores
  """
  scores = [scoring_function(solution, pred) for pred in predictions]
  return scores

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
    default_time_budget = 1200
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

    logger.debug("Version: {}. Description: {}".format(VERSION, DESCRIPTION))
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
    ingestion_start = ingestion_info['start_time']
    logger.debug("Ingestion start time: {}".format(ingestion_start))
    logger.debug("Scoring start time: {}".format(scoring_start))
    # Get ingestion PID
    ingestion_pid = ingestion_info['ingestion_pid']
    # Update time_budget if exists
    if 'time_budget' in ingestion_info:
      if not time_budget == ingestion_info['time_budget']:
        logger.warning("Different time budgets passed to ingestion and " +
                       "scoring. Use that of ingestion.")
        time_budget = ingestion_info['time_budget']

    # Get the metric
    scoring_function = autodl_auc
    metric_name = "Area under Learning Curve"

    # Get all the solution
    solution = get_solution(solution_dir)
    # Check if the task is multilabel (i.e. with one hot label)
    is_multiclass_task = is_multiclass(solution)
    # Extract the dataset name from the file name
    basename = get_task_name(solution_dir)

    scoring_success = True
    time_limit_exceeded = False

    try:
      # Begin scoring process, along with ingestion program
      # Moniter training processes while time budget is not attained
      prediction_files_so_far = []
      scores_so_far = []
      num_preds = 0
      while(time.time() < ingestion_start + time_budget):
        if not ingestion_is_alive(prediction_dir):
          logger.info("Detected ingestion program had stopped running " +
                      "because an 'end.txt' file is written by ingestion. " +
                      "Stop scoring now.")
          break
        time.sleep(1)
        # Get list of prediction files
        prediction_files = get_prediction_files(prediction_dir)
        num_preds_new = len(prediction_files)
        if(num_preds_new > num_preds):
          new_prediction_files = get_new_prediction_files(prediction_dir,
                                                prediction_files_so_far)
          new_scores = [scoring_function(solution, read_array(pred))
                        for pred in new_prediction_files]
          prediction_files_so_far += new_prediction_files
          logger.info("[+] New prediction found. Now number of predictions " +
                       "made = {}".format(num_preds_new))
          prediction_files_so_far += new_prediction_files
          scores_so_far += new_scores
          score = update_score_and_learning_curve(prediction_dir,
                                                  basename,
                                                  ingestion_start,
                                                  solution_dir,
                                                  scoring_function,
                                                  score_dir,
                                                  is_multiclass_task)
          num_preds = num_preds_new
          logger.info("Current area under learning curve for {}: {:.4f}"\
                    .format(basename, score))
      else: # When time budget is used up, kill ingestion
        if is_process_alive(ingestion_pid):
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
                                            solution_dir,
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

# Author: Zhengying LIU
# Creation date: 20 Sep 2018
"""This script allows participants to run local test of their method within the
downloaded starting kit folder (and avoid using submission quota on CodaLab). To
do this, run:
```
python run_local_test.py -dataset_dir='./AutoDL_sample_data/' -code_dir='./AutoDL_sample_code_submission/'
```
in the starting kit directory. If you want to test the performance of a
different algorithm on a different dataset, please specify them using respective
arguments (flags).

If you want to use default folders (i.e. those in above command line), simply
run
```
python run_local_test.py
```
"""

import tensorflow as tf
import os
import time
import webbrowser
from multiprocessing import Process

def _HERE(*args):
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(h, *args)

def get_path_to_ingestion_program(starting_kit_dir):
  return os.path.join(starting_kit_dir,
                      'AutoDL_ingestion_program', 'ingestion.py')

def get_path_to_scoring_program(starting_kit_dir):
  return os.path.join(starting_kit_dir,
                      'AutoDL_scoring_program', 'score.py')


def run_baseline(dataset_dir, code_dir):
    # Current directory containing this script
    starting_kit_dir = os.path.dirname(os.path.realpath(__file__))
    path_ingestion = get_path_to_ingestion_program(starting_kit_dir)
    path_scoring = get_path_to_scoring_program(starting_kit_dir)

    # Run ingestion and scoring at the same time
    command_ingestion = 'python {} {} {}'.format(path_ingestion, dataset_dir, code_dir)
    command_scoring = 'python {} {}'.format(path_scoring, dataset_dir)
    def run_ingestion():
      os.system(command_ingestion)
    def run_scoring():
      os.system(command_scoring)
    ingestion_process = Process(name='ingestion', target=run_ingestion)
    scoring_process = Process(name='scoring', target=run_scoring)
    ingestion_process.start()
    scoring_process.start()
    detailed_results_page = os.path.join(starting_kit_dir,
                                         'AutoDL_scoring_output',
                                         'detailed_results.html')
    detailed_results_page = os.path.abspath(detailed_results_page)

    # Open detailed results page in a browser
    time.sleep(2)
    for i in range(30):
      if os.path.isfile(detailed_results_page):
        webbrowser.open('file://'+detailed_results_page, new=2)
        break
      time.sleep(1)


if __name__ == '__main__':
    default_starting_kit_dir = os.path.abspath(os.path.join(_HERE()))
    # The default dataset is 'miniciao' under the folder AutoDL_sample_data/
    default_dataset_dir = os.path.join(default_starting_kit_dir,
                                       'AutoDL_sample_data', 'miniciao')
    default_code_dir = os.path.join(default_starting_kit_dir,
                                       'AutoDL_sample_code_submission')

    tf.flags.DEFINE_string('dataset_dir', default_dataset_dir,
                          "Directory containing the content (e.g. adult.data/ + "
                          "adult.solution) of an AutoDL dataset. Specify this "
                          "argument if you want to test on a different dataset.")

    tf.flags.DEFINE_string('code_dir', default_code_dir,
                          "Directory containing a `model.py` file. Specify this "
                          "argument if you want to test on a different algorithm.")

    FLAGS = tf.flags.FLAGS
    dataset_dir = FLAGS.dataset_dir
    code_dir = FLAGS.code_dir
    run_baseline(dataset_dir, code_dir)

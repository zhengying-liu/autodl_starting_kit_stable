#!/usr/bin/python
# Author: Liu Zhengying
# Date: 10 Apr 2019
# Description: This script download the 5 datasets used in AutoCV challenge and
#   put them under the folder AutoDL_sample_data/. This script supports
#   breakpoint resume, which means that you can recover downloading from where
#   your network broke down.

import os
import sys

def main(*argv):
  dataset_names = ['Munster', 'Chucky','Pedro', 'Decal', 'Hammer']
  data_urls ={
      'Munster':'https://autodl.lri.fr/my/datasets/download/6662aa6e-75ab-439c-bf98-97dd11401053',
      'Chucky':'https://autodl.lri.fr/my/datasets/download/d06aa5fc-1fb5-4283-8e05-abed4ccdd975',
      'Pedro':'https://autodl.lri.fr/my/datasets/download/61a074cd-e909-4d49-b313-7da0d4f7dc8b',
      'Decal':'https://autodl.lri.fr/my/datasets/download/dfd93c39-e0d4-41b2-b332-4dd002676e05',
      'Hammer':''
  }
  solution_urls = {
      'Munster':'https://autodl.lri.fr/my/datasets/download/f3a61a40-b1f1-4ded-bc55-fb730a12f4c4',
      'Chucky':'https://autodl.lri.fr/my/datasets/download/29932707-21cc-4670-a7db-cdc246a8ab71',
      'Pedro':'https://autodl.lri.fr/my/datasets/download/852c1e68-5e91-477e-bef0-824b503814e8',
      'Decal':'https://autodl.lri.fr/my/datasets/download/d72cba79-3051-4779-b624-e50335aad874',
      'Hammer':'https://autodl.lri.fr/my/datasets/download/e5a392c6-3bd1-4acf-8dcc-77bf89448616'
  }
  for dataset_name in dataset_names:
    msg = "Downloading data files and solution file for the dataset {}..."\
          .format(dataset_name)
    le = len(msg)
    print('\n' + '#'*(le+10))
    print('#'*4+' ' + msg + ' '+'#'*4)
    print('#'*(le+10) + '\n')
    data_url = data_urls[dataset_name]
    solution_url = solution_urls[dataset_name]
    dataset_dir = "AutoDL_sample_data/" + dataset_name
    os.system('mkdir -p {}'.format(dataset_dir))
    data_zip_file = "AutoDL_sample_data/{}/{}.data.zip"\
                    .format(dataset_name, dataset_name)
    solution_zip_file = "AutoDL_sample_data/{}/{}.solution.zip"\
                        .format(dataset_name, dataset_name)
    os.system('wget -q --show-progress -c -N {} -O {}'\
              .format(data_url, data_zip_file))
    os.system('wget -q --show-progress -c -N {} -O {}'\
              .format(solution_url, solution_zip_file))
    os.system('unzip -n -d AutoDL_sample_data/{} {}'\
              .format(dataset_name, data_zip_file))
    os.system('unzip -n -d AutoDL_sample_data/{} {}'\
              .format(dataset_name, solution_zip_file))
  print("\nFinished downloading 5 public datasets: 'Munster', 'Chucky','Pedro', 'Decal', 'Hammer'.")
  print("Now you should find them under the directory 'AutoDL_sample_data/'.")

if __name__ == '__main__':
  main(sys.argv)

#!/usr/bin/python
# Author: Liu Zhengying
# Date: 10 Apr 2019
# Description: This script downloads the 5 datasets used in AutoCV challenge and
#   put them under the folder AutoDL_public_data/. This script supports
#   breakpoint resume, which means that you can recover downloading from where
#   your network broke down.

import os
import sys

def main(*argv):
  dataset_names = ['Munster', 'Chucky','Pedro', 'Decal',
                   'Hammer', 'Kreatur', 'Katze', 'Kraut']
  data_urls ={
      'Munster':'https://autodl.lri.fr/my/datasets/download/6662aa6e-75ab-439c-bf98-97dd11401053',
      'Chucky':'https://autodl.lri.fr/my/datasets/download/d06aa5fc-1fb5-4283-8e05-abed4ccdd975',
      'Pedro':'https://autodl.lri.fr/my/datasets/download/61a074cd-e909-4d49-b313-7da0d4f7dc8b',
      'Decal':'https://autodl.lri.fr/my/datasets/download/dfd93c39-e0d4-41b2-b332-4dd002676e05',
      'Hammer':'https://autodl.lri.fr/my/datasets/download/eb569948-72f0-4002-8e4d-479a27766cbf',
      'Kreatur':'https://autodl.lri.fr/my/datasets/download/9dd0fded-7c58-4768-805d-875e82191043',
      'Katze':'https://autodl.lri.fr/my/datasets/download/fc318649-5224-458c-bb17-0d89b76a75dd',
      'Kraut':'https://autodl.lri.fr/my/datasets/download/39614747-95bd-49fb-aded-2437dd1675ba'
  }
  solution_urls = {
      'Munster':'https://autodl.lri.fr/my/datasets/download/f3a61a40-b1f1-4ded-bc55-fb730a12f4c4',
      'Chucky':'https://autodl.lri.fr/my/datasets/download/29932707-21cc-4670-a7db-cdc246a8ab71',
      'Pedro':'https://autodl.lri.fr/my/datasets/download/852c1e68-5e91-477e-bef0-824b503814e8',
      'Decal':'https://autodl.lri.fr/my/datasets/download/d72cba79-3051-4779-b624-e50335aad874',
      'Hammer':'https://autodl.lri.fr/my/datasets/download/c3729c98-4755-47a2-b764-a4159c5ca152',
      'Kreatur':'https://autodl.lri.fr/my/datasets/download/d79782d6-f967-4883-b417-1206566dd69c',
      'Katze':'https://autodl.lri.fr/my/datasets/download/a0713a0d-1214-427c-b67e-302d4e3efe0c',
      'Kraut':'https://autodl.lri.fr/my/datasets/download/0e1e5166-40a5-4daa-a053-8bb506fac5a1'
  }

  def _HERE(*args):
      h = os.path.dirname(os.path.realpath(__file__))
      return os.path.abspath(os.path.join(h, *args))
  starting_kit_dir = _HERE()
  public_date_dir = os.path.join(starting_kit_dir, 'AutoDL_public_data')

  for dataset_name in dataset_names:
    msg = "Downloading data files and solution file for the dataset {}..."\
          .format(dataset_name)
    le = len(msg)
    print('\n' + '#'*(le+10))
    print('#'*4+' ' + msg + ' '+'#'*4)
    print('#'*(le+10) + '\n')
    data_url = data_urls[dataset_name]
    solution_url = solution_urls[dataset_name]
    dataset_dir = os.path.join(public_date_dir, dataset_name)
    os.system('mkdir -p {}'.format(dataset_dir))
    data_zip_file = os.path.join(dataset_dir, dataset_name + '.data.zip')
    solution_zip_file = os.path.join(dataset_dir,
                                     dataset_name + '.solution.zip')
    os.system('wget -q --show-progress -c -N {} -O {}'\
              .format(data_url, data_zip_file))
    os.system('wget -q --show-progress -c -N {} -O {}'\
              .format(solution_url, solution_zip_file))
    os.system('unzip -n -d {} {}'\
              .format(dataset_dir, data_zip_file))
    os.system('unzip -n -d {} {}'\
              .format(dataset_dir, solution_zip_file))
  print("\nFinished downloading {} public datasets: {}"\
        .format(len(dataset_names),dataset_names))
  print("Now you should find them under the directory: {}"\
        .format(public_date_dir))

if __name__ == '__main__':
  main(sys.argv)

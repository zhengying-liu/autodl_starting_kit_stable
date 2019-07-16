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
      'Munster':'https://autodl.lri.fr/my/datasets/download/d29000a6-b5b8-4ccf-9050-2686af874a71',
      'Chucky':'https://autodl.lri.fr/my/datasets/download/cf2176c2-5454-4d07-9c4e-758e3c5bcb31',
      'Pedro':'https://autodl.lri.fr/my/datasets/download/d556ca67-01c7-4a8d-9a74-a2bd9c06414d',
      'Decal':'https://autodl.lri.fr/my/datasets/download/31a34c03-a75c-4e0f-b72d-3723ba303dac',
      'Hammer':'https://autodl.lri.fr/my/datasets/download/3507841e-59fe-4598-a27e-a9e170b26e44',
      'Kraut':'https://autodl.lri.fr/my/datasets/download/a1d9f706-cf8d-4a63-a544-552d6b85d6c4',
      'Katze':'https://autodl.lri.fr/my/datasets/download/611a42fa-da42-4a30-8c7a-69230d9eeb92',
      'Kreatur':'https://autodl.lri.fr/my/datasets/download/c240df57-b144-41df-a059-05bc859d2621'      
  }
  solution_urls = {
      'Munster':'https://autodl.lri.fr/my/datasets/download/5a24d8f3-dfb6-4935-b798-14baccda695f',
      'Chucky':'https://autodl.lri.fr/my/datasets/download/ba4837bf-275d-43a6-a481-d03dce7ba127',
      'Pedro':'https://autodl.lri.fr/my/datasets/download/9993ea27-955e-4faa-9d28-4a7dfe1fcc55',
      'Decal':'https://autodl.lri.fr/my/datasets/download/cc93c74c-2732-4e7d-ae7f-a2c3bc555360',
      'Hammer':'https://autodl.lri.fr/my/datasets/download/e5b6188f-a377-4a5d-bbe1-a586716af487',
      'Kraut':'https://autodl.lri.fr/my/datasets/download/47ff016d-cc66-47a9-945d-bc01fd9096c9',
      'Katze':'https://autodl.lri.fr/my/datasets/download/a04de92e-b04b-49a6-96c2-5910c64f9b3c',
      'Kreatur':'https://autodl.lri.fr/my/datasets/download/31ecdb19-c25a-420f-9764-8d1783705deb'
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

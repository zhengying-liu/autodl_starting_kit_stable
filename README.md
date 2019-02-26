AutoDL starting kit
======================================

ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
UNIVERSITE PARIS SUD, INRIA, CHALEARN, AND/OR OTHER ORGANIZERS
OR CODE AUTHORS DISCLAIM ANY EXPRESSED OR IMPLIED WARRANTIES.

## Download This Starting Kit

 You can download this starting kit by clicking on the green button "Clone or download" on top of [this GitHub repo](https://github.com/zhengying-liu/autodl_starting_kit_stable), then "Download ZIP". You'll have this whole starting kit by unzipping the downloaded file.

 Another convenient way is to use **git clone**:
 ```
 cd <path_to_your_directory>
 git clone https://github.com/zhengying-liu/autodl_starting_kit_stable.git
 ```

 Then you can begin your participation to AutoDL challenge by carefully reading this README.md file (which you are already doing).

## Local Development and Testing
To make your own submission to AutoDL challenge, you need to modify the file
`model.py` in `AutoDL_sample_code_submission/`, which implements the logic of your
algorithm. You can then test it in the exact same environment as the CodaLab
environment using Docker. *WARNING*: You can choose to run local test out of the Docker
image, but it's possible that certain Python packages you use are not installed
in the Docker image used in the competition.

If you are new to docker, install docker from https://docs.docker.com/get-started/.
Then, at the shell, run:
```
cd path/to/AutoDL_starting_kit/
docker run --memory=4g -it -u root -v $(pwd):/app/codalab evariste/autodl:dockerfile bash
```
You will then be able to run the ingestion program (to produce predictions) and
the scoring program (to evaluate your predictions) on toy sample data. In AutoDL
challenge, these two programs will run in parallel to give real-time feedback
(with learning curves). So we provide a Python script to simulate this behavior:
```
python run_local_test.py
```
Then you can view the real-time feedback with a learning curve by opening the
HTML page in `AutoDL_scoring_output/`.

The full usage is
```
python run_local_test.py -dataset_dir='./AutoDL_sample_data/' -code_dir='./AutoDL_sample_code_submission/'
```
You can change the argument `dataset_dir` to other AutoDL datasets (e.g. those
you downloaded from **Get Data** section of the challenge). On the other hand,
you can also modify the directory containing your other sample code
(`model.py`).

WARNING: when you run local test in a Docker container, **make sure you distribute
enough RAM** (at least 4GB). Otherwise, it's possible that certain task
(especially when the dataset is large) will get 'Killed'. You can modify memory
allocation of Docker in 'Preferences -> Advanced'.

## How to prepare a ZIP file for submission
Zip the contents of AutoDL_sample_code_submission (without the directory structure)
```
zip mysubmission.zip AutoDL_sample_code_submission/*
```
and use the "Upload a Submission" button for make a submission to CodaLab.

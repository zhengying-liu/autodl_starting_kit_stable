AutoDL starting kit
======================================

ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
UNIVERSITE PARIS SUD, INRIA, CHALEARN, AND/OR OTHER ORGANIZERS
OR CODE AUTHORS DISCLAIM ANY EXPRESSED OR IMPLIED WARRANTIES.

## Download this starting kit

 You can download this starting kit by clicking on the green button "Clone or download" on top of [this GitHub repo](https://github.com/zhengying-liu/autodl_starting_kit_stable), then "Download ZIP". You'll have this whole starting kit by unzipping the downloaded file.

 Another convenient way is to use **git clone**:
 ```
 cd <path_to_your_directory>
 git clone https://github.com/zhengying-liu/autodl_starting_kit_stable.git
 ```

Then you can begin participating to the AutoDL challenge by carefully reading this README.md file.

## Local development and testing
To make your own submission to AutoDL challenge, you need to modify the file
`model.py` in `AutoDL_sample_code_submission/`, which implements the logic of your
algorithm. You can then test it on your local computer using Docker, in the exact same environment as on the CodaLab challenge plarform. Advanced users can also run local test without Docker, if they install all the required packages, see the [Docker file](https://github.com/zhengying-liu/autodl/blob/master/docker/Dockerfile).

If you are new to docker, install docker from https://docs.docker.com/get-started/.
Then, at the shell, run:
```
cd path_to/AutoDL_starting_kit_stable/
docker run --memory=4g -it -u root -v "$(pwd):/app/codalab" -p 8888:8888 evariste/autodl
```
Make sure you use enough RAM (**at least 4GB**). If the port 8888 is occupied,
you can use other ports, e.g. 8899, and use instead the option `-p 8899:8888`.

You will then be able to run the `ingestion program` (to produce predictions) and
the `scoring program` (to evaluate your predictions) on toy sample data. In the AutoDL
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

## How to run the tutorial
We provide a tutorial in the form of a Jupyter notebook. When you are in your docker container, enter:
```
jupyter-notebook --ip=0.0.0.0 --allow-root &
```
Then cut and paste the URL containing your token. It should look like something like that:
```
http://0.0.0.0:8888/?token=82e416e792c8f6a9f2194d2f4dbbd3660ad4ca29a4c58fe7
```
and select README.ipynb in the menu.

## How to prepare a ZIP file for submission
Zip the contents of `AutoDL_sample_code_submission`(or any folder containing your `model.py` file) without the directory structure:
```
cd AutoDL_sample_code_submission/
zip -r mysubmission.zip *
```
then use the "Upload a Submission" button to make a submission to CodaLab.

Tip: to look at what's in your submission zip file without unzipping it, you
can do
```
unzip -l mysubmission.zip
```

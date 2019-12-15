# Song Genre and Hit Prediction

This repository contains the files of the project Song Genre and Hit Prediction. The structure of this repository with all the files are present in the `ee660_project_dirtree` document. 

### Instructions to execute the code

1. Run the `download_models.sh`script to download the models.
2. Execute the `main.py` file with optional arguments such as to construct the dataset or to train the models.
- The `main.py` file calls the appropriate file for performing the respective function. For example, if the user input an argument for constructing the dataset, then the `dataset_constructor.py` file is executed.
- Same is the case for the file `train.py` which trains the models with the dataset constructed in the previous step
3. The primary function of `main.py` is to run the trained models downloaded in step 1 on the validation data to perform model selection based on their performance. To do this, the `main.py` file executes `validation.py` which evaluates and selects the best model based on the evaluation.
4. The objective is to see how well a model can generalize its predictions, and this is done by `test.py` file which the `main.py` file executes to evaluate the models on unseen out of sample data.

### Example execution commands
Step 1:
```sh
$ bash download_models.sh
```
Step 2:
To construct the dataset:
```sh
$ python3 main.py -c True
```
To train the models on previously constructed datasets:
```sh
$ python3 main.py -t True
```
To construct the dataset and train the models on the newly constructed datasets:
```sh
$ python3 main.py -c True -t True
```
Step 3:
To validate the models and choose the best model and evaluate its performance:
```sh
$ python3 main.py
```

The evaluation results will be displayed on the command line directly, and the intermediate and final results with the evaluation metrics will be stored in the `results` directory.

Additionally the constructed datasets before feature extraction are provided in the `data_files.tar.gz` compressed file

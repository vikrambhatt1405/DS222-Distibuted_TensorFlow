# DS222 Distributed Tensorflow Models ![CI status](https://img.shields.io/badge/build-passing-brightgreen.svg)

## Assignment Report
**Assignment2** folder contains all the codes for local and distributed settings.

## Directories

**obj** contains all the training,test,validation features extracted using doc2vec model and dictionaries in .pkl format which contains map from unique labels to index.

**tf_logs** contains logs in local training which can be visualized with tensorbaord.

**synchronous_sgd** contains python scritps which can be run on cluster for synchrnous training by specifying appropriate tensorflow logs locations and ip address and ports of the nodes.For example, to start a worker job with task index 1 at 
node with ip *10.24.1.211* port 2000 run the following command.

```
python sync_sgd.py --ps_hosts=10.24.1.210:2000 --worker_hosts=10.24.1.211:2001,10.24.1.212:2002 --job_name=worker --task_index=1
```

**asyncronous_sgd** contains python scripts for asynchronous training which can be run same way as desribed above.For example to start a parameter server with task index 0 on port 2000 at node with ip address *10.24.1.210* run the following command.

```
python sync_sgd.py --ps_hosts=10.24.1.210:2000 --worker_hosts=10.24.1.211:2001,10.24.1.212:2002 --job_name=ps --task_index=0
```
---

## Notebooks
* I have made the following pipeline for data preprocessing and model training.

* For feature extraction and label extraction use FeatureExtraction.ipynb and LabelExtraction.ipynb.
* These notebook train the doc2vec model on the given dataset and extract labels for each training example and store them in a dictiionary which contains mappings from indices to unqiue number of labels seen in training set.

* For data preparation and preprocessing use DataPrepration.ipynb. This notebook create folder obj/ which stores training,test and development features in numpy format(.npy) which can be easily loaded further down the pipeline.

* For the training model in local setting look into logistic_regression.ipynb which runs on GPU.

* For adaptive learning rate training of the same model, look into Adaptive_learning_rate.ipynb

* All the tensorflow logs and summary files can be specified in their respective notebook can be visualized through tensorboar

## Data
* All the data from clustern and local setting has been gathered from tensorbaord logs in .csv format, can be found in **data** folder.

## Datasets
* A small sample of datasets used for training and testing can be found in **DBPedia.verysmall** folder.

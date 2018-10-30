#Assignment Report

This folder contains all the codes for local and distributed settings.

##Directories

\obj contains all the training,test,validation features extracted using doc2vec model and dictionaries in .pkl format which contains map from unique labels to index.

\tf_logs contains logs in local training which can be visualized with tensorbaord.

\synchronous_sgd contains python scritps which can be run on cluster for synchrnous training by specifying appropriate tensorflow logs locations and ip address and ports of the nodes.An example

'''
python sync_sgd.py --ps_hosts=10.24.1.210:2000 --worker_hosts=10.24.1.211:2001,10.24.1.212:2002 --job_name=worker --task_index=1
'''

##Notebooks

I have made the following pipeline for data preprocessing and model training.
For feature extraction and label extraction use FeatureExtraction.ipynb and LabelExtraction.ipynb.
These notebook train the doc2vec model on the given dataset and extract labels for each training example and store them in a dictiionary which contains mappings from indices to unqiue number of labels seen in training set.
For data preparation and preprocessing use DataPrepration.ipynb. This notebook create folder obj/ which stores training,test and development features in numpy format(.npy) which can be easily loaded further down the pipeline.
For the training model in local setting look into logistic_regression.ipynb which runs on GPU.
For adaptive learning rate training of the same model, look into Adaptive_learning_rate.ipynb
All the tensorflow logs and summary files can be specified in their respective notebook can be visualized through tensorboard.


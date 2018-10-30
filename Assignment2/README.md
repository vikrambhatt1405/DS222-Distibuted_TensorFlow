This folder contains all the codes for local and distributed settings.
I have made the following pipeline for data preprocessing and model training.
For feature extraction and label extraction use FeatureExtraction.ipynb and LabelExtraction.ipynb.
These notebook train the doc2vec model on the given dataset and extract labels for each training example and store them in a dictiionary which contains mappings from indices to unqiue number of labels seen in training set.
For data preparation and preprocessing use DataPrepration.ipynb. This notebook create folder obj/ which stores training,test and development features in numpy format(.npy) which can be easily loaded further down the pipeline.
For the training model in local setting look into logistic_regression.ipynb which runs on GPU.
For adaptive learning rate training of the same model, look into Adaptive_learning_rate.ipynb
All the tensorflow logs and summary files can be specified in their respective notebook can be visualized through tensorboard.


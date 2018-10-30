import argparse
import sys
import numpy as np
import pickle
import os
import tensorflow as tf
from datetime import datetime

def logistic_regression(X, y,n_classes,initializer=None,seed=42,learning_rate=0.01):
    n_inputs_including_bias = int(X.shape[1])
    beta=0.01
    with tf.name_scope("logistic_regression"):
        with tf.name_scope("model"):
            if initializer is None:
                initializer = tf.random_uniform([n_inputs_including_bias, n_classes], -1.0, 1.0, seed=seed)
            theta = tf.Variable(initializer, name="theta")
            logits = tf.matmul(X, theta, name="logits")
        with tf.name_scope("train"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=y, logits=logits)+beta*tf.nn.l2_loss(theta))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            loss_summary = tf.summary.scalar('Cross_Entropy_Loss', loss)
        with tf.name_scope("init"):
            init = tf.global_variables_initializer()
        with tf.name_scope("save"):
            saver = tf.train.Saver()
    return loss,optimizer,loss_summary, init, saver,logits

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch

def add_biastomatrix(*args):
    matrices=[]
    for X in args:
        m=X.shape[0]
        X=np.c_[np.ones((m,1)),X]
        matrices.append(X)
    return tuple(matrices)

def calculateScores(logits,y):
    max_scores_indices=np.argmax(logits,axis=1)
    n_examples=logits.shape[0]
    predictions=np.zeros_like(logits,dtype=np.int)
    predictions[range(n_examples),max_scores_indices]=1
    del logits
    accuracy=accuracy_score(y,predictions)
    precision=precision_score(y,predictions,average="macro")
    recall=recall_score(y,predictions,average='macro')
    return accuracy,precision,recall

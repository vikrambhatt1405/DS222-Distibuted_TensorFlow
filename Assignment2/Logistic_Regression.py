
# coding: utf-8

# In[29]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
import pickle
import os
from sklearn.metrics import recall_score,precision_score,accuracy_score
get_ipython().run_line_magic('precision', '3')


# In[2]:


def logistic_regression(X, y,n_classes,initializer=None,seed=42,learning_rate=0.01):
    n_inputs_including_bias = int(X.shape[1])
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
            training_op = optimizer.minimize(loss)
            loss_summary = tf.summary.scalar('Cross-Entropy_Loss', loss)
        with tf.name_scope("init"):
            init = tf.global_variables_initializer()
        with tf.name_scope("save"):
            saver = tf.train.Saver()
    return loss, training_op, loss_summary, init, saver,logits


# In[3]:


from datetime import datetime

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


# In[4]:


X_train=np.load("obj/X_train.npy")
y_train=np.load("obj/y_train.npy")
X_test=np.load("obj/X_test.npy")
y_test=np.load("obj/y_test.npy")
X_devel=np.load("obj/X_devel.npy")
y_devel=np.load("obj/y_devel.npy")


# In[5]:


def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch


# In[6]:


def add_biastomatrix(*args):
    matrices=[]
    for X in args:
        m=X.shape[0]
        X=np.c_[np.ones((m,1)),X]
        matrices.append(X)
    return tuple(matrices)


# In[7]:


(X_train_withbias,X_test_withbias,X_devel_withbias)=add_biastomatrix(X_train,X_test,X_devel)


# In[8]:


print("Train data with bias shape:{}".format(X_train_withbias.shape))
print("Test data with bias shape:{}".format(X_test_withbias.shape))
print("Development data with bias shape:{}".format(X_devel_withbias.shape))


# In[9]:


beta=0.01
n_inputs = 100
logdir = log_dir("logreg")
n_classes=y_train.shape[1]
X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, n_classes), name="y")
loss, training_op, loss_summary, init, saver,logits = logistic_regression(X, y,n_classes)
train_writer = tf.summary.FileWriter(logdir+"/train", tf.get_default_graph())
test_writer=tf.summary.FileWriter(logdir+"/test")
devel_writer=tf.summary.FileWriter(logdir+"/devel")
merged=tf.summary.merge_all()


# In[24]:


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


# In[26]:


m=X_train.shape[0]
n_epochs = 1001
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
checkpoint_path = "/tmp/logistic_regressionl2.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./my_logreg_model"

with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):
        # if the checkpoint file exists, restore the model and load the epoch number
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print("Training was interrupted. Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)

    for epoch in range(start_epoch, n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train_withbias, y_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        test_loss,summary_test = sess.run([loss,merged], feed_dict={X: X_test_withbias, y: y_test})
        test_writer.add_summary(summary_test, epoch)
        devel_loss,summary_devel=sess.run([loss,merged],feed_dict={X:X_devel_withbias,y:y_test})
        devel_writer.add_summary(summary_devel,epoch)
        if epoch % 100 == 0:
            train_loss,summary_train=sess.run([loss,merged],feed_dict={X:X_train_withbias,y:y_train})
            train_writer.add_summary(summary_train,epoch)
            print("Epoch:", epoch, "\tTest Loss:", test_loss,"\t Devel Loss:",
                  devel_loss,"\tTrain Loss:",train_loss)
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))

    saver.save(sess, final_model_path)
    os.remove(checkpoint_epoch_path)
    test_logits=sess.run(logits,feed_dict={X:X_test_withbias})
    devel_logits=sess.run(logits,feed_dict={X:X_devel_withbias})
    (test_accuracy,test_precision,test_recall)=calculateScores(np.array(test_logits),y_test)
    (devel_accuracy,devel_precision,devel_recall)=calculateScores(np.array(devel_logits),y_devel)
    print("Test Set Scores\n")
    print("Test Accuracy:",test_accuracy,"Test Precision:",test_precision,"Test Recall:",test_recall,"\n")
    print("Development Set Scores\n")
    print("Dev Accuracy:",devel_accuracy,"Dev Precision:",devel_precision,"Dev Recall:",devel_recall)

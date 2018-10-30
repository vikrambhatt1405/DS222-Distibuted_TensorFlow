import tensorflow as tf
import numpy as np
from helper_functions import logistic_regression
from helper_functions import add_biastomatrix
from helper_functions import calculateScores
from helper_functions import random_batch
#from sklearn.utils import shuffle
REPLICAS_TO_AGGREGATE = 2

# Here we define our cluster setup via the command line
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Define the characteristics of the cluster node, and its task index
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS


def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")


  # Create a cluster following the command line paramaters.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":


    is_chief = (FLAGS.task_index == 0) #checks if this is the chief node

    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      #Define the training set, and the model parameters, loss function and training operation
      #trX = np.linspace(-1, 1, 101)
      X_train=np.load("obj/X_train.npy")
      y_train=np.load("obj/y_train.npy")
      X_test=np.load("obj/X_test.npy")
      y_test=np.load("obj/y_test.npy")
      X_devel=np.load("obj/X_devel.npy")
      y_devel=np.load("obj/y_devel.npy")
      (X_train_withbias,X_test_withbias,X_devel_withbias)=add_biastomatrix(X_train,X_test,X_devel)
      #trY = 2 * trX + np.random.randn(*trX.shape) * 0.4 + 0.2 # create a y value
      logdir="/home/vikrambhatt/"
      beta=0.01
      n_inputs = 100
      n_classes=y_train.shape[1]
      X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
      y = tf.placeholder(tf.float32, shape=(None, n_classes), name="y")
      loss, _ , loss_summary,_, saver,_ = logistic_regression(X, y,n_classes)
      merged=tf.summary.merge_all()
      global_step = tf.train.get_or_create_global_step()
      optimizer=tf.train.GradientDescentOptimizer(.0001)
      sync_optimizer=tf.train.SyncReplicasOptimizer(optimizer,
        replicas_to_aggregate=REPLICAS_TO_AGGREGATE,total_num_replicas=2)
      opt=sync_optimizer.minimize(loss,global_step=global_step)

      sync_replicas_hook=sync_optimizer.make_session_run_hook(is_chief)
      summary_hook = tf.train.SummarySaverHook(save_steps=5,output_dir="/home/vikrambhatt/tf_summary_synclogs_worker{}".format(FLAGS.task_index),summary_op=merged)
      checkpoint_hook = tf.train.CheckpointSaverHook(save_steps=5,checkpoint_dir="/home/vikrambhatt/tf_synccheckpoint_logs_worker{}".format(FLAGS.task_index),saver=saver)

      hooks=[sync_replicas_hook,tf.train.StopAtStepHook(last_step=10),summary_hook,checkpoint_hook]

      train_writer = tf.summary.FileWriter(logdir+"tf_summary_synclogs_worker{}/train".format(FLAGS.task_index), tf.get_default_graph())
      test_writer=tf.summary.FileWriter(logdir+"tf_summary_synclogs_worker{}/test".format(FLAGS.task_index))
      devel_writer=tf.summary.FileWriter(logdir+"tf_summary_synclogs_worker{}/devel".format(FLAGS.task_index))
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    # Create a "supervisor", which oversees the training process.
    sess = tf.train.MonitoredTrainingSession(master=server.target,
				is_chief=(FLAGS.task_index == 0),
                  		hooks=hooks,
				save_summaries_steps=5)
    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
      # Loop until the supervisor shuts down
    step = 0
    n_epochs=1000
    m=X_train.shape[0]
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))
    while not sess.should_stop() :
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
	start_epoch=1
	for epoch in range(start_epoch, n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = random_batch(X_train_withbias, y_train, batch_size)
            _, step = sess.run([opt,global_step], feed_dict={X: X_batch, y: y_batch})
            test_loss,summary_test = sess.run([loss,merged], feed_dict={X: X_test_withbias, y: y_test})
            devel_loss,summary_devel=sess.run([loss,merged],feed_dict={X:X_devel_withbias,y:y_devel})
            test_writer.add_summary(summary_test, epoch)
	    devel_writer.add_summary(summary_test, epoch)
            if epoch % 10 == 0:
            	test_loss,summary_test = sess.run([loss,merged], feed_dict={X: X_test_withbias, y: y_test})
            	devel_loss,summary_devel=sess.run([loss,merged],feed_dict={X:X_devel_withbias,y:y_devel})
                train_loss,summary_train=sess.run([loss,merged],feed_dict={X:X_train_withbias,y:y_train})
                test_writer.add_summary(summary_test, epoch)
                devel_writer.add_summary(summary_test, epoch)

                print("From Node:"+str(FLAGS.task_index)+"\n")
		print("Epoch:", epoch, " Test Loss:", test_loss," Devel Loss:",devel_loss," Train Loss:",train_loss)

    print("Maximum epochs reached.Exiting the session from worker {}".format(FLAGS.task_index))

    sess.stop()

if __name__ == "__main__":
  tf.app.run()

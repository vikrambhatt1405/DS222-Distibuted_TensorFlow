`'''
192.168.1.7$ python example.py --job-name="ps" --task_index=0 
192.168.1.2$ python example.py --job-name="worker" --task_index=0 
192.168.1.8$ python example.py --job-name="worker" --task_index=1 
192.168.1.9$ python example.py --job-name="worker" --task_index=2 

reference from ischlag.github.io
'''
from __future__ import print_function
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os import path
import tensorflow as tf
import numpy as np
import argparse
import time
import sys
import gzip

#cluster specification
parameter_servers = ["192.168.1.7:2223"]
workers = ["192.168.1.2:2223",
           "192.168.1.8:2223",
           "192.168.1.9:2223"]

cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_string(
    "log_dir", "lastest_train_log/sound_classifier/", """Log directory to view tensorboard""")
tf.app.flags.DEFINE_integer(
    "batch_size", 1024, """batch size to train [default:1024]""")
tf.app.flags.DEFINE_integer("epoch", 10, """epoch size [default:10]""")
tf.app.flags.DEFINE_integer(
    "test_timing_point", 512, """set a testing interval [default:512]""")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
if FLAGS.job_name == "ps":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    server = tf.train.Server(cluster, 
                        job_name=FLAGS.job_name,
                        task_index=FLAGS.task_index)
else:
    server = tf.train.Server(cluster, 
                        job_name=FLAGS.job_name,
                        task_index=FLAGS.task_index)
#input config
num_input = 13
num_classes = 3
h_layer_units = 512
number_of_h_layer = 6
test_timing_point = 512
learning_rate = 1E-4

if FLAGS.job_name == "ps":
    print("calling join to serve")
    server.join()
elif FLAGS.job_name == "worker":

#-----function definition-------
    def extract_data_set(filename, batch_size, shuffle_batch=False,name=None):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)
        record_default = [[1], [1.0], [1.0], [1.0], [1.0], [1.0], [
            1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
        labels, mfcc1, mfcc2, mfcc3, mfcc4, mfcc5, mfcc6, mfcc7, mfcc8, mfcc9, mfcc10, mfcc11, mfcc12, mfcc13 = tf.decode_csv(
            value, record_defaults=record_default)
        mfcc_features = tf.stack([mfcc1, mfcc2, mfcc3, mfcc4, mfcc5,
                                mfcc6, mfcc7, mfcc8, mfcc9, mfcc10, mfcc11, mfcc12, mfcc13])
        labels = tf.one_hot(labels,depth=3,dtype=tf.int32,name="labels_hot")
        labels_batch, mfcc_features_batch = tf.train.batch(
            [labels, mfcc_features], batch_size=batch_size, allow_smaller_final_batch=True)
        return labels_batch, mfcc_features_batch

    def hidden_layer(input, size_in, size_out, name="hidden_layer",reuse=False):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([size_in,size_out],stddev=0.1),name="W")
            b = tf.Variable(tf.constant(0.1,shape=[size_out]),name="B")
            act = tf.nn.tanh(tf.add(tf.matmul(input, w), b))
            tf.summary.histogram("weights", w)
            tf.summary.histogram("bias", b)
            tf.summary.histogram("activations", act)
            return act


    def fully_connected_layer(input, size_in, size_out, name="fc",reuse=False):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal(shape=[size_in,size_out],stddev=0.1),name="W")
            b = tf.Variable(tf.constant(0.1,shape=[size_out]),name="B")
            act = tf.add(tf.matmul(input, w), b)
            tf.summary.histogram("weights", w)
            tf.summary.histogram("bias", b)
            tf.summary.histogram("activations", act)
            return act
#-------start program------------
    print("calling worker")
    with tf.device("/job:ps/task:0"):
        print("Calculating data size...")
        with gzip.open('csv_files/mfcc_combine_train_shuffled.gz') as f:
            text = f.readlines()
            size1 = len(text)
            print("train:{}".format(size1))
        with gzip.open('csv_files/mfcc_combine_test_shuffled.gz') as f:
            text = f.readlines()
            size2 = len(text)
            print("train:{}".format(size2))


        del text
        test_timing = (size2 / FLAGS.batch_size) / \
            (size1 / FLAGS.batch_size / FLAGS.test_timing_point)
        print("test timing:{}".format(test_timing))
        steps = size1 / FLAGS.batch_size
        print("steps:{}".format(steps))
    with tf.device("/job:ps/task:0"):
        train_labels_batch, train_mfcc_batch = extract_data_set(
            "csv_files/mfcc_combine_train_shuffled.csv", FLAGS.batch_size,"train")
        test_labels_batch, test_mfcc_batch = extract_data_set(
            "csv_files/mfcc_combine_test_shuffled.csv", FLAGS.batch_size,"test")

    # def sound_classifier_model(learning_rate, number_of_h_layer, h_layer_units, hparam):
        # tf.reset_default_graph()
    #saver = tf.train.Saver(sharded=True)
    
    # Between-graph replication model
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" %FLAGS.task_index,cluster=cluster)):
        is_chief = (FLAGS.task_index == 0)
        global_step = tf.get_variable('global_step', [], 
                                      initializer = tf.constant_initializer(0),dtype=tf.int32)
        # global_step = tf.Variable(0,name='global_step')
        #global_step=tf.get_variable('global_step', shape=[], initializer=tf.zeros_initializer(), dtype=tf.int32, trainable=False)
        x = tf.placeholder(tf.float32,shape=[None,num_input],name="mfcc_input")
        y = tf.placeholder(tf.int32,shape=[None,num_classes],name="labels")
        is_training = tf.placeholder(dtype=bool, shape=(), name="is_training")
        q_selector = tf.cond(is_training,lambda:[train_mfcc_batch,train_labels_batch],lambda:[test_mfcc_batch,test_labels_batch])

        h = []
        h.append(hidden_layer(x,num_input,h_layer_units))
        for i in range(number_of_h_layer):
            h.append(hidden_layer(h[i],h_layer_units,h_layer_units))

        output = fully_connected_layer(h[number_of_h_layer],h_layer_units,num_classes)
        saver = tf.train.Saver()
        with tf.name_scope("x-entropy"):
            xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                    logits=output, labels=y), name="x-entropy")

        with tf.name_scope("train"):
            grad_op = tf.train.AdamOptimizer(learning_rate)
            rep_op = tf.train.SyncReplicasOptimizer(grad_op,
                                                    replicas_to_aggregate=len(workers),
                                                    total_num_replicas=len(workers))
            train_op = rep_op.minimize(xent,global_step=global_step)
            sync_replicas_hook = rep_op.make_session_run_hook(is_chief)
            # grads = rep_op.compute_gradients(xent)
            # apply_gradients_op = rep_op.apply_gradients(grads,global_step=global_step)
            # with tf.control_dependencies([apply_gradients_op]):
                # train_op=tf.identity(xent,name='train_op')
            # train_step = rep_op.minimize(xent,global_step=global_step,aggregation_method=tf.AggregationMethod.ADD_N)
        init_token_op = rep_op.get_init_tokens_op()
        chief_queue_runner = rep_op.get_chief_queue_runner()
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        tf.summary.scalar("x-entropy", xent)
        tf.summary.scalar("accuracy",accuracy)
        
        summary_op = tf.summary.merge_all()
        global_init_op = tf.global_variables_initializer()
        local_init_op = tf.local_variables_initializer()
        
    #------------prepare for session------------
    # coord = tf.train.Coordinator()
    print("creating Supervisor...")
    # sv = tf.train.Supervisor(is_chief=is_cheif,
    #                         global_step=global_step,
    #                         saver = saver,
    #                         logdir=FLAGS.log_dir)
    print("making config...")
    sess_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True,device_filters=["/job:ps","/job:worker/task:%d"%FLAGS.task_index])
    begin_time = time.time()
    summary_hook = tf.train.SummarySaverHook(save_secs=600,output_dir=FLAGS.log_dir,summary_op=summary_op)
    checkpoint_hook = tf.train.CheckpointSaverHook(save_steps=test_timing,checkpoint_dir=FLAGS.log_dir,saver=saver)

    #---------------training session--------------
    print("waiting for session prepare....")
    with tf.train.MonitoredTrainingSession(server.target,is_chief=is_chief,hooks=[sync_replicas_hook,summary_hook,checkpoint_hook],config=sess_config) as sess:
        while not sess.should_stop():
            '''
            # is cheif
            if FLAGS.task_index == 0:
                sv.start_queue_runners(sess,[cheif_queue_runner])
                sess.run(init_token_op)
            '''
            print("in session")
            # while not sv.should_stop():
                # coord = tf.train.Coordinator()
            # sess.run(tf.global_variables_initializer())
            # sess.run(tf.local_variables_initializer())
            # coord = tf.train.Coordinator()
            # if FLAGS.task_index == 0:
            tf.train.start_queue_runners(sess=sess)
             sess.run(init_token_op)
            # sv.start_queue_runners(sess=sess)
            with tf.device("/job:ps/task:0/cpu:0"):
                f = open(FLAGS.log_dir + "accuracy_" + str(learning_rate) +
                        "_with_" + str(number_of_h_layer) + "layers", "w+")
                print("loading data to queue....")
            
            print("done loading to queue")
            print("start adding summary")
            with tf.device("/job:ps/task:0/cpu:0"):
                f = open(FLAGS.log_dir+"accuracy_"+str(learning_rate)+"_with_"+str(number_of_h_layer)+"layers","w+")

                # writer = tf.summary.FileWriter(FLAGS.log_dir,graph=tf.get_default_graph())
                # writer = tf.summary.FileWriter(FLAGS.log_dir + "lr_%.0E layers:%d"%(learning_rate,6))
                # writer.add_graph(sess.graph)
                #perform training cycles
                start_time = time.time()

            print("start training")
            for i in range(steps):
                print("read data")
                mfcc_batch,label_batch = sess.run(q_selector,feed_dict={is_training:True})
                # label_batch = sess.run(tf.one_hot(y_batch,depth=3,dtype=tf.int32,name="labels"))
                [_,loss,glob_step,train_accuracy] = sess.run([train_op,xent,global_step,accuracy],feed_dict={x:mfcc_batch,y:label_batch})
                # [train_accuracy,s] = sess.run([accuracy,summary_op],feed_dict={x:mfcc_batch,y:label_batch})
                print("done")
                elapsed = round(time.time()-start_time,2)
                sys.stdout.write("\r")
                sys.stdout.write("learning_rate:{0},{1} layers , training:{2}/{3} , train_accuracy:{4} [elapsed_time:{5:.2f}] ".format(learning_rate,number_of_h_layer,i,steps,train_accuracy,elapsed))
                sys.stdout.flush()
                # writer.add_summary(summary,i)
                print("before train")
                # sess.run(global_step,init_token_op,chief_queue_runner)
                # sess.run(train_op,xent)
                print("after train")
                if i != 0 and i % test_timing == 0:
                    # saver.save(sess, os.path.join(FLAGS.log_dir, "model.ckpt"), i)
                    test_time += 1
                    for j in range(test_timing):
                        mfcc_batch,y_batch = sess.run(q_selector,feed_dict={is_training:False})
                        test_label = sess.run(tf.one_hot(y_batch,depth=3,dtype=tf.int32,name="test_label"))
                        test_accuracy = accuracy.eval(feed_dict={x:mfcc_batch,y:test_label})
                        sys.stdout.write("\r")
                        sys.stdout.write("{}layers: test_num:{} testing:{} test_accuracy:{}".format(number_of_h_layer,test_time,j,test_accuracy))
                        sys.stdout.flush()
                        with tf.device("/job:ps/task:0/cpu:0"):
                            f.write("{}layers: test_num:{} testing:{} test_accuracy:{}\n".format(number_of_h_layer,test_time,j,test_accuracy))
                            f.flush()

    f.write("total_training for lr {},{}layers time:{} minutes".format(learning_rate,number_of_h_layer,((time.time()-start_time)/60.0)))
    f.flush()
    f.close()
    print("-----%s minutes ------"%((time.time()-start_time)/60.0))
#    sv.stop()
    print("Done!")
`

import input_data
import sys
from input_data import create_datasets
import numpy as np
import tensorflow as tf
import time

# maximal number of images for each class, can be modified
NUM_IMAGES = 1000

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def relu_weight_variable(shape):
    assert len(shape) is 2
    input_size = shape[0]
    initial = tf.truncated_normal(shape, stddev=np.sqrt(2.0 / input_size))
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, strides):
    return conv_batch_normalization(tf.nn.conv2d(x, W, strides=strides, padding='SAME'))

def conv_batch_normalization(x):
    mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
    return tf.nn.batch_normalization(x, mean, variance, None, None, 0.0001)

def fc_batch_normalization(x):
    mean, variance = tf.nn.moments(x, axes=[0])
    return tf.nn.batch_normalization(x, mean, variance, None, None, 0.0001)

# host name for ps and workers, split by ','
# one may have multipel parameter servers and multiple workers
parameter_servers = ["sgs-gpu-01:2222"]
workers = ["sgs-gpu-02:2222", "sgs-gpu-03:2222"]
#workers = ["spaceml1:2222"]

cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
# in this example, the input should be like:
# python train.py --job_name="ps" --task_index=0 [on sgs-gpu-01]
# python train.py --job_name="worker" --task_index=0 [on sgs-gpu-02]
# python train.py --job_name="worker" --task_index=1 [on sgs-gpu-03]

tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
# every task_index corresponds to a machine 

server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)


list_ = []
for line in open("/home/litian/data/label.txt"):
    list_.append(['a', line.strip('\n')])
classes = np.array(list_)
print len(classes)


train_dataset, val_dataset, test_dataset = create_datasets(classes[:, 1], num_samples=NUM_IMAGES, val_fraction=0.2,
                                                           test_fraction=0.2)
val_images, val_labels = val_dataset.next_batch(20)

num_classes = len(classes)
print num_classes

# if it is a parameter-server, waiting for the updated parameters from other workers
if FLAGS.job_name == "ps":
    server.join()

# if it is a worker who is responsible for computing, go on with calculating gradiants and updated values
# if using GPUs, delete "/cpu:0" in "/job:worker/task:%d/cpu:0"
# if using CPUs, keep it and specify "export CUDA_VISIBLE_DEVICES=" in the terminal before "python train.py ..."
elif FLAGS.job_name == "worker":
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d/cpu:0" % FLAGS.task_index,
        cluster=cluster)):
    # all the 'parameters' defined under "with ...replica_device_setter" 
    # will be maintained in the parameter-sserver

    # define a model as usual

        x = tf.placeholder(tf.float32, shape=[None, input_data.IMAGE_WIDTH * input_data.IMAGE_HEIGHT, 3])
        y_ = tf.placeholder(tf.float32, shape=[None, num_classes])
        keep_prob = tf.placeholder(tf.float32)
    
        x_reshaped = tf.reshape(x, [-1, input_data.IMAGE_WIDTH, input_data.IMAGE_HEIGHT, 3])  
        #First convolutional layer, (224, 224, 3) to (56, 56, 96)    
        W_conv1 = weight_variable([11, 11, 3, 96])    
        b_conv1 = bias_variable([96]) # convert it to (56,56,96) now     
        h_conv1 = tf.nn.relu(conv2d(x_reshaped, W_conv1, [1, 4, 4, 1]) + b_conv1)   
        print h_conv1.get_shape()
    
        #max_pool1 = tf.nn.max_pool(h_conv1, ksize = [1,3,3,1], strides = [1,2,2,1], padding='SAME'     
        # (56,56,96)->(28,28,96)
        norm1 = tf.nn.lrn(h_conv1, 5, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)   
        max_pool1 = tf.nn.max_pool(norm1, ksize = [1,3,3,1], strides = [1,2,2,1], padding='SAME')
        print max_pool1.get_shape()   
        #h_conv1 = tf.nn.relu(conv2d(x_reshaped, W_conv1, [1, 1, 1, 1]) + b_conv1 # 
        # Second convolutional layer, (28,28,96) to (28, 28, 256) to (14,14,256)    
        W_conv2 = weight_variable([5, 5, 96, 256])  
        b_conv2 = bias_variable([256])     
        h_conv2 = tf.nn.relu(conv2d(max_pool1, W_conv2, [1, 1, 1, 1]) + b_conv2) 
        print h_conv2.get_shape()   
        #h_pool2 = tf.nn.max_pool(h_conv2, ksize = [1,3,3,1], strides = [1,2,2,1], padding='SAME'     
        norm2 = tf.nn.lrn(h_conv2, 5, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)    
        h_pool2 = tf.nn.max_pool(norm2, ksize = [1,3,3,1], strides = [1,2,2,1], padding='SAME') # 
        print h_pool2.get_shape()
    
        # Third convolutional layer, (14,14,256) to (14, 14, 384)     
        W_conv3 = weight_variable([3, 3, 256, 384])    
        b_conv3 = bias_variable([384])     
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, [1, 1, 1, 1]) + b_conv3)
        print h_conv3.get_shape()
    
        # # Fourth convolutional layer, (14, 14, 384) to (14, 14, 384)     
        W_conv4 = weight_variable([3, 3, 384, 384])    
        b_conv4 = bias_variable([384])     
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, [1, 1, 1, 1]) + b_conv4) # 
        print h_conv4.get_shape()
    
        # Fifth convolutional layer, (14, 14, 384) to (7, 7, 256)     
        W_conv5 = weight_variable([3, 3, 384, 256])    
        b_conv5 = bias_variable([256])     
        h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, [1, 1, 1, 1]) + b_conv5)     
        max_pooling5 = tf.nn.max_pool(h_conv5, ksize = [1,3,3,1], strides = [1,2,2,1], padding='SAME') # 
        print max_pooling5.get_shape()
    
        # First fully-connected laye     
        W_fc1 = relu_weight_variable([7*7*256, 4096])
        b_fc1 = bias_variable([4096])     
        h_conv5_flat = tf.reshape(max_pooling5, [-1, 7*7*256])  
        print h_conv5_flat.get_shape()   
        h_fc1 = tf.nn.relu(fc_batch_normalization(tf.matmul(h_conv5_flat, W_fc1) + b_fc1))     
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # 
        # Second fully-connected laye     
        W_fc2 = relu_weight_variable([4096,4096])
        b_fc2 = bias_variable([4096])     
        h_fc2 = tf.nn.relu(fc_batch_normalization(tf.matmul(h_fc1_drop, W_fc2) + b_fc2))   
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob) # 
        # Third fully-connected laye     
        W_fc3 = relu_weight_variable([4096, num_classes])    
        b_fc3 = bias_variable([num_classes])     
        y_score = fc_batch_normalization(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)   
        y_logit = tf.nn.softmax(y_score)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_score, labels=y_))

        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    
        correct_prediction = tf.equal(tf.argmax(y_logit, 1), tf.argmax(y_, 1))
    
        y_max = tf.reduce_min(tf.reduce_max(y_logit,1))
        y_label_max = tf.reduce_max(y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()     
        init_op = tf.global_variables_initializer()


    # use the worker with index_0 as the main machine to save the models and initialize the parameters in a distributed setting
    # other workers wait during this period

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/mnt/ds3lab/litian/logs",
                             init_op=init_op,
                             saver=saver)

    with sv.managed_session(server.target) as sess:

        for i in range(200000):
            image_batch, label_batch = train_dataset.next_batch(60, random_crop=True)
            sess.run(train_step, feed_dict={x: image_batch, y_: label_batch, keep_prob: 0.5})
    
            if i % 5 == 0:
                train_accuracy = sess.run(accuracy,feed_dict={x: image_batch, y_: label_batch, keep_prob: 1.0})
                train_cost, y_2,y_3, y_4 = sess.run([cross_entropy,y_max, y_label_max, y_logit], feed_dict={x: image_batch, y_: label_batch, keep_prob: 1.0})
                localtime = time.asctime(time.localtime(time.time()))
                print localtime
                assert y_3 > 0
	      
	   #
	   #print tf.shape(y_4)
                print("step %d, training accuracy %g, training loss %g" % (i, train_accuracy, train_cost))
    
            if i % 25 == 0:
                val_accuracy = sess.run(accuracy, feed_dict={x: val_images, y_: val_labels, keep_prob: 1.0})
            #val_writer.add_summary(summary, i)
                print("validation set accuracy %g" % val_accuracy)

    sv.stop()










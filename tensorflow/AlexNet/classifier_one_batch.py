import input_data
import sys
from input_data_one_batch import create_datasets
import numpy as np
import tensorflow as tf
import time

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


list_ = []
for line in open("/home/litian/data/label.txt"):
    list_.append(['a', line.strip('\n')])
classes = np.array(list_)
print len(classes)


train_dataset, val_dataset, test_dataset = create_datasets(classes[:, 1], num_samples=NUM_IMAGES, val_fraction=0.2,
                                                           test_fraction=0.2)
num_classes = len(classes)
print num_classes

# replace with GPU
with tf.device('/gpu:0'):
# Placeholders
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
    #y_logit = create_net(x_reshaped, keep_prob, num_classes, input_data.IMAGE_HEIGHT)
# # Training
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_score, labels=y_))
# #tf.scalar_summary('xentropy', cross_entropy)
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    train_step2 = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_logit, 1), tf.argmax(y_, 1))
    y_max = tf.reduce_min(tf.reduce_max(y_logit,1))
    y_label_max = tf.reduce_max(y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#tf.scalar_summary('accuracy', accuracy)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config = config)
sess.run(tf.initialize_all_variables())
#sess = tf.Session()
#merged = tf.merge_all_summaries()
#train_writer = tf.train.SummaryWriter(summaries_dir + '/train', sess.graph)
#val_writer = tf.train.SummaryWriter(summaries_dir + '/val')
#sess.run(tf.initialize_all_variables()) # need to use tf.global_variables_initializer instead

val_images, val_labels = val_dataset.next_batch(20)

for i in range(200000):
    image_batch, label_batch = train_dataset.next_batch(90, random_crop=True)
    if ( i * 180 > 4000 * 100):
        sess.run(train_step, feed_dict={x: image_batch, y_: label_batch, keep_prob: 0.5})
    else:
        sess.run(train_step2, feed_dict={x: image_batch, y_: label_batch, keep_prob: 0.5})
    if i % 5 == 0:
        train_accuracy = sess.run(accuracy,feed_dict={x: image_batch, y_: label_batch, keep_prob: 1.0})
        #train_writer.add_summary(summary, i)
        train_cost, y_2,y_3, y_4 = sess.run([cross_entropy,y_max, y_label_max, y_logit], feed_dict={x: image_batch, y_: label_batch, keep_prob: 1.0})
        localtime = time.asctime(time.localtime(time.time()))
        print localtime
	assert y_3 > 0
	print y_2, y_3
	print tf.size(y_4)
	print tf.shape(y_4)
        print("step %d, training accuracy %g, cost %g" % (i, train_accuracy, train_cost))

    if i % 25 == 0:
        val_accuracy = sess.run(accuracy, feed_dict={
            x: val_images, y_: val_labels, keep_prob: 1.0})
        #val_writer.add_summary(summary, i)
        print("validation set accuracy %g" % val_accuracy)

import input_data
import sys
from input_data import create_datasets
import numpy as np
import tensorflow as tf
import time

NUM_IMAGES = 1000


list_ = []
for line in open("/home/litian/data/label.txt"):
    list_.append(['a', line.strip('\n')])
classes = np.array(list_)
print len(classes)


train_dataset, val_dataset, test_dataset = create_datasets(classes[:, 1], num_samples=NUM_IMAGES, val_fraction=0.05,
                                                           test_fraction=0.05)
num_classes = len(classes)
print num_classes

with tf.device('/gpu:1'): 
    x = tf.placeholder(tf.float32, shape=[None, input_data.IMAGE_WIDTH * input_data.IMAGE_HEIGHT, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes])
    keep_prob = tf.placeholder(tf.float32)
    x_reshaped = tf.reshape(x, [-1, input_data.IMAGE_WIDTH, input_data.IMAGE_HEIGHT, 3]) 
    # conv1_1
    kernel1_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1))
    conv1_1 = tf.nn.conv2d(x_reshaped, kernel1_1, [1, 1, 1, 1], padding='SAME')
    biases1_1 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True)
    out1_1 = tf.nn.bias_add(conv1_1, biases1_1)
    relu1_1 = tf.nn.relu(out1_1)

    # conv1_2    
    kernel1_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1))
    conv1_2 = tf.nn.conv2d(relu1_1, kernel1_2, [1, 1, 1, 1], padding='SAME')
    biases1_2 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True)
    out1_2 = tf.nn.bias_add(conv1_2, biases1_2)
    relu1_2 = tf.nn.relu(out1_2)

    # pool1
    pool1 = tf.nn.max_pool(relu1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    # conv2_1
    kernel2_1 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                             stddev=1e-1))
    conv2_1 = tf.nn.conv2d(pool1, kernel2_1, [1, 1, 1, 1], padding='SAME')
    biases2_1 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=True)
    out2_1 = tf.nn.bias_add(conv2_1, biases2_1)
    relu2_1 = tf.nn.relu(out2_1)

  #conv2_2
    kernel2_2 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                             stddev=1e-1))
    conv2_2 = tf.nn.conv2d(relu2_1, kernel2_2, [1, 1, 1, 1], padding='SAME')
    biases2_2 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=True)
    out2_2 = tf.nn.bias_add(conv2_2, biases2_2)
    relu2_2 = tf.nn.relu(out2_2)

    # pool2
    pool2 = tf.nn.max_pool(relu2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    # conv3_1
    kernel3_1 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1))
    conv3_1 = tf.nn.conv2d(pool2, kernel3_1, [1, 1, 1, 1], padding='SAME')
    biases3_1 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True)
    out3_1 = tf.nn.bias_add(conv3_1, biases3_1)
    relu3_1 = tf.nn.relu(out3_1)

    # conv3_2
    kernel3_2 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                             stddev=1e-1))
    conv3_2 = tf.nn.conv2d(relu3_1, kernel3_2, [1, 1, 1, 1], padding='SAME')
    biases3_2 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True)
    out3_2 = tf.nn.bias_add(conv3_2, biases3_2)
    relu3_2 = tf.nn.relu(out3_2)
        # conv3_3
    kernel3_3 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                             stddev=1e-1))
    conv3_3 = tf.nn.conv2d(relu3_2, kernel3_3, [1, 1, 1, 1], padding='SAME')
    biases3_3 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True)
    out3_3 = tf.nn.bias_add(conv3_3, biases3_3)
    relu3_3 = tf.nn.relu(out3_3)

        # pool3
    pool3 = tf.nn.max_pool(relu3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

        # conv4_1
    kernel4_1 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                             stddev=1e-1))
    conv4_1 = tf.nn.conv2d(pool3, kernel4_1,[1, 1, 1, 1], padding='SAME')
    biases4_1 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True)
    out4_1 = tf.nn.bias_add(conv4_1, biases4_1)
    relu4_1 = tf.nn.relu(out4_1)

        # conv4_2
    kernel4_2 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1))
    conv4_2 = tf.nn.conv2d(relu4_1, kernel4_2, [1, 1, 1, 1], padding='SAME')
    biases4_2 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True)
    out4_2 = tf.nn.bias_add(conv4_2, biases4_2)
    relu4_2 = tf.nn.relu(out4_2)

        #conv4_3
    kernel4_3 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1))
    conv4_3 = tf.nn.conv2d(relu4_2, kernel4_3, [1, 1, 1, 1], padding='SAME')
    biases4_3 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True)
    out4_3 = tf.nn.bias_add(conv4_3, biases4_3)
    relu4_3 = tf.nn.relu(out4_3)

        # pool4
    pool4 = tf.nn.max_pool(relu4_3,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')

    # conv5_1
    kernel5_1 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1))
    conv5_1 = tf.nn.conv2d(pool4, kernel5_1, [1, 1, 1, 1], padding='SAME')
    biases5_1 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True)
    out5_1 = tf.nn.bias_add(conv5_1, biases5_1)
    relu5_1 = tf.nn.relu(out5_1)

    # conv5_2
    kernel5_2 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1))
    conv5_2 = tf.nn.conv2d(relu5_1, kernel5_2, [1, 1, 1, 1], padding='SAME')
    biases5_2 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True)
    out5_2 = tf.nn.bias_add(conv5_2, biases5_2)
    relu5_2 = tf.nn.relu(out5_2)

        # conv5_3
    kernel5_3 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1))
    conv5_3 = tf.nn.conv2d(relu5_2, kernel5_3, [1, 1, 1, 1], padding='SAME')
    biases5_3 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True)
    out5_3 = tf.nn.bias_add(conv5_3, biases5_3)
    relu5_3 = tf.nn.relu(out5_3)

        # pool5
    pool5 = tf.nn.max_pool(relu5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    #def fc_layers(self):
        # fc1
    shape = int(np.prod(pool5.get_shape()[1:]))
    fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                 dtype=tf.float32,
                                                 stddev=1e-1))
    fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                         trainable=True)
    pool5_flat = tf.reshape(pool5, [-1, shape])
    fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
    fc1 = tf.nn.relu(fc1l)
        # fc2
    fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                 dtype=tf.float32,
                                                 stddev=1e-1))
    fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                         trainable=True)
    fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
    fc2 = tf.nn.relu(fc2l)

        # fc3
    fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                 dtype=tf.float32,
                                                 stddev=1e-1))
    fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),trainable=True)
    fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)

    y_logit = tf.nn.softmax(fc3l)

# # Training
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_logit, labels=y_))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_logit, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables()) # need to use tf.global_variables_initializer instead

val_images, val_labels = val_dataset.next_batch(200)

for i in range(200000):
    image_batch, label_batch = train_dataset.next_batch(256, random_crop=True)
    sess.run(train_step, feed_dict={x: image_batch, y_: label_batch, keep_prob: 0.5})
    # original paper: keep_prob: 0.5; batch size: 256
    if i % 5 == 0:
        train_accuracy = sess.run(accuracy,feed_dict={x: image_batch, y_: label_batch, keep_prob: 1.0})
        train_cost = sess.run(cross_entropy, feed_dict={x: image_batch, y_: label_batch, keep_prob: 1.0})
        localtime = time.asctime(time.localtime(time.time()))
        print localtime
        print("step %d, training accuracy %g, cost %g" % (i, train_accuracy, train_cost))

    if i % 25 == 0:
        val_accuracy = sess.run(accuracy, feed_dict={
            x: val_images, y_: val_labels, keep_prob: 1.0})
        print("validation set accuracy %g" % val_accuracy)
import input_data
import sys
from input_data import create_datasets
import numpy as np
import tensorflow as tf
import time
from inception_v1 import inception_v1

NUM_IMAGES = 1000


list_ = []
for line in open("/home/litian/data/label.txt"):
    list_.append(['a', line.strip('\n')])
classes = np.array(list_)
print len(classes)


train_dataset, val_dataset, test_dataset = create_datasets(classes[:, 1], num_samples=NUM_IMAGES, val_fraction=0.05,
                                                           test_fraction=0.05)
num_classes = len(classes) # should be 1000
print num_classes

with tf.device('/gpu:1'): 
    x = tf.placeholder(tf.float32, shape=[None, input_data.IMAGE_WIDTH * input_data.IMAGE_HEIGHT, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes])
    keep_prob = tf.placeholder(tf.float32)
    x_reshaped = tf.reshape(x, [-1, input_data.IMAGE_WIDTH, input_data.IMAGE_HEIGHT, 3]) 
    
    y_logit = inception_v1(x_reshaped,1000)
    print ("y_prediction's shape is:")
    print tf.shape(y_logit)

    print ("label y's shape is")
    print tf.shape(y_)

# # Training
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_logit, labels=y_))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    print tf.shape(tf.argmax(y_logit, 1))
    print tf.shape(tf.argmax(y_, 1))
    correct_prediction = tf.equal(tf.argmax(y_logit, 1), tf.argmax(y_, 1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables()) # need to use tf.global_variables_initializer instead

val_images, val_labels = val_dataset.next_batch(200)

for i in range(200000):
    image_batch, label_batch = train_dataset.next_batch(256, random_crop=True)
    sess.run(train_step, feed_dict={x: image_batch, y_: label_batch, keep_prob: 0.5})
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
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

# Read MNIST data set (Train data from CSV file)
data = pd.read_csv('./kaggle/train.csv')
# Next Batch

def next_batch1(batch_size):    
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        epochs_completed += 1
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]




########Functions

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# Extracting images and labels from given data
# For images
images = data.iloc[:,1:].values
images = images.astype(np.float)

# For labels
labels_flat = data[[0]].values.ravel()
labels_count = np.unique(labels_flat).shape[0]


labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

## Data Extraction end

# Normalize from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)

# Split data into training & validation
train_images = images[:]
train_labels = labels[:]

#train = DataSet(train_images, train_labels,reshape=False)

#validation = DataSet(validation_images, validation_labels, reshape=False)


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


x_image = tf.reshape(x, [-1, 28, 28, 1])


#Conv1

W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)


W_conv1_1 = weight_variable([3, 3, 32, 32])
b_conv1_1 = bias_variable([32])
h_conv1_1 = tf.nn.relu(conv2d(h_conv1, W_conv1_1) + b_conv1_1)

#Convpool2
W_conv1_2 = weight_variable([3, 3, 32, 32])
b_conv1_2 = bias_variable([32])
h_conv1_2 = tf.nn.relu(conv2d(h_conv1_1, W_conv1_2) + b_conv1_2)

h_pool1 = max_pool_2x2(h_conv1_2)


#Convpool3
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

W_conv2_1 = weight_variable([3, 3, 64, 64])
b_conv2_1 = bias_variable([64])
h_conv2_1 = tf.nn.relu(conv2d(h_conv2, W_conv2_1) + b_conv2_1)


W_conv2_2 = weight_variable([3, 3, 64, 64])
b_conv2_2 = bias_variable([64])
h_conv2_2 = tf.nn.relu(conv2d(h_conv2_1, W_conv2_2) + b_conv2_2)

h_pool2 = max_pool_2x2(h_conv2_2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])


#fully connected
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


#Adding dropout to fully connected layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])


#Output
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

Y_pred = y_conv

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
predict = tf.argmax(Y_pred, 1)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

BATCH_SIZE = 100
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]
# visualisation variables
train_accuracies = []
validation_accuracies = []

for i in range(10000):
  batch = next_batch1(BATCH_SIZE)
  if i%100 == 0:

#	sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
#    train_accuracy = accuracy.eval(feed_dict={
#        x:batch[0], y_: batch[1], keep_prob: 1.0})
    train_accuracy = sess.run(accuracy, feed_dict = {x : batch[0], y_:batch[1], keep_prob: 1.0})	
    print("step %d, training accuracy %g"%(i, train_accuracy))
    train_accuracies.append(train_accuracy)	
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


#test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

#print("test accuracy on mnist test images%g" %test_accuracy)
test_acc = 0;
for i in range(100):
	testSet = mnist.test.next_batch(50)
	test_acc += (accuracy.eval(feed_dict={ x: testSet[0], y_: testSet[1], keep_prob: 1.0}))
	print("test accuracy %g"%accuracy.eval(feed_dict={ x: testSet[0], y_: testSet[1], keep_prob: 1.0}))
	
print("Test Accuracy Total:%.4f"%test_acc);
# print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# read test data from CSV file 
test_data = pd.read_csv('./kaggle/test.csv')
test_labels_flat = test_data[[0]].values.ravel()
test_labels_count = np.unique(test_labels_flat).shape[0]

test_images = test_data.values
test_images = test_images.astype(np.float)
test_labels = dense_to_one_hot(test_labels_flat, test_labels_count)
test_labels = test_labels.astype(np.uint8)

# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)

print('test_images({0[0]},{0[1]})'.format(test_images.shape))


# predict test set
#predicted_lables = predict.eval(feed_dict={X: test_images, keep_prob: 1.0})
# using batches is more resource efficient
predicted_lables = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]//BATCH_SIZE):
    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE], keep_prob: 1.0})


# save results
np.savetxt('submission_7layercnn_mod.csv', np.c_[range(1,len(test_images)+1),predicted_lables], delimiter=',', header = 'imageid,label', comments = '', fmt='%d')

sess.close()








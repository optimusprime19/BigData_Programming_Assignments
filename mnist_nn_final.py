#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

# Read MNIST data set (Train data from CSV file)
data = pd.read_csv('./kaggle/train.csv')



################# Dataset class
class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=False, dtype=dtypes.float32, reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

################### End #####################33

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

train = DataSet(train_images, train_labels,reshape=False)

#validation = DataSet(validation_images, validation_labels, reshape=False)


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

drop_conv = tf.placeholder('float')
drop_hidden = tf.placeholder('float')

#sess = tf.Session()

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

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


# visualisation variables
train_accuracies = []
validation_accuracies = []

for i in range(10000):
  batch = train.next_batch(50)
  if i%100 == 0:

#	sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
#    train_accuracy = accuracy.eval(feed_dict={
#        x:batch[0], y_: batch[1], keep_prob: 1.0})
    train_accuracy = sess.run(accuracy, feed_dict = {x : batch[0], y_:batch[1], keep_prob: 1.0})	
    print("step %d, training accuracy %g"%(i, train_accuracy))
    train_accuracies.append(train_accuracy)	
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


#test_accuracy = sess.run(accuracy, feed_dict={
#    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

#print("test accuracy %g" %test_accuracy)


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
BATCH_SIZE = 100
# using batches is more resource efficient
predicted_lables = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]//BATCH_SIZE):
    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE], keep_prob: 1.0})


# save results
np.savetxt('submission.csv', np.c_[range(1,len(test_images)+1),predicted_lables], delimiter=',', header = 'imageid,label', comments = '', fmt='%d')

sess.close()








#Importing Packages
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.cmap'] = 'Greys'

import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision = 2)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print (mnist.train.images.shape)
print (mnist.train.labels.shape)

print (mnist.test.images.shape)
print (mnist.test.labels.shape)

example_image = mnist.train.images[1]
example_image_reshaped = example_image.reshape((28,28))
example_label = mnist.train.labels[1]

print (example_label)
plt.imshow(example_image_reshaped)

#Spliting the data into bataches makes it easier on memory to train

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.truncated_normal(shape=[784,400], stddev = 0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[400]))
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal(shape=[400,100], stddev = 0.1))
b2= tf.Variable(tf.constant(0.1, shape=[100]))
h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

W3 = tf.Variable(tf.truncated_normal(shape=[100,10], stddev = 0.1))
b3= tf.Variable(tf.constant(0.1, shape=[10]))
y = tf.nn.softmax(tf.matmul(h2, W3) + b3)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


#intialize the session

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#print out results

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

# -*- coding: utf-8 -*-
"""
**********************************************************************
* Filename    : mnist_basic_two_layer.py
* Description : A basic two layer neral network of the mnist dataset
* Created on  : Mon Mar  6 11:44:54 2017
* @author     : haidyn
**********************************************************************
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import sys

mnist = input_data.read_data_sets("/tmp/data")

# how many different output classes are there in the data
n_classes = len(np.unique(mnist.train.labels))
# how many input features in each image
n_features = np.shape(mnist.train.images)[1] 
# number of training examples
n_train = np.shape(mnist.train.images)[0]
# number of validation examples
n_valid = np.shape(mnist.validation.images)[0]

# display some random sample images to verify the input images are as we expect
random_num = np.random.randint(0, n_train)
# get a random image and reshape the data to a 28 x 28 tensor
image = mnist.train.images[random_num]
image = image.reshape((28,28))

plt.title("Number %d" % (mnist.train.labels[random_num]))
plt.axis('off')
plt.imshow(image, cmap=cm.Greys)

# define a set structure for each linear transform layer
# also used on the final output layer
def linear_layer(x_data, node_count):
    # x_data: Input features from the precedding layer
    # node_count: number of nodes outputing to in the next layer
    # return: activated output of current layer
    input_features = x_data.get_shape().as_list()[1] # number of features in x_data

    w = tf.Variable(tf.truncated_normal(shape=(input_features, node_count), mean=0, stddev=0.1))
    b = tf.Variable(tf.zeros(node_count))
    
    return tf.add(tf.matmul(x_data, w), b)

# define a set structure for each hidden layer
def layer(x_data, node_count):
    # x_data: Input features from the precedding layer
    # node_count: number of nodes outputing to in the next layer
    # return: activated output of current layer
    linear = linear_layer(x_data, node_count)
    # activate the layer
    relu = tf.nn.relu(linear)
    return relu

# create the neural model for learning the data
def model(x):
    # Layer 1: Input features = 784, Output = 100
    lay1 = layer(x, 100)
    # Layer 2: Input features = 100, Output = 10
    lay2 = linear_layer(lay1, n_classes)
    return lay2

# creates a one hot encoded vectors for the training data
def one_hot_encode(labels):
    # labels: label position data to be one hot encoded, [3, ...]
    # return a tensor with one hot encoding of data of labels, [[0,0,0,1,0],[...]]
    depth = len(np.unique(labels))
    size = np.shape(labels)[0]
    one_hot = np.zeros((size, depth))
    one_hot[np.arange(size), labels] = 1
    return one_hot


# define the system placeholders
x = tf.placeholder(tf.float32, (None, n_features))
y = tf.placeholder(tf.int16, (None, n_classes))

# one hot encode the training, validation and test data
one_hot_train = one_hot_encode(mnist.train.labels)
one_hot_valid = one_hot_encode(mnist.validation.labels)
one_hot_test = one_hot_encode(mnist.test.labels)

# Hyper parameters
epochs = 50
learn_rate = 0.01
batch_size = 128

# run the model
logits = model(x)
# get the final output of the model and find the loss 
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(cross_entropy)
# Update the model to predict better results based of the training loss
optimizer = tf.train.AdamOptimizer(learn_rate)
# train the model to minimise the loss
training = optimizer.minimize(loss)

# Accuracy of the model measured against the validation set
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Traing the model
    print("Training")
    for epoch in range(epochs):
        # shuffle the training data to prevent similar batches
        x_train, y_train = shuffle(mnist.train.images, one_hot_train)
        
        # train with ALL the training data per epoch, Training each pass with 
        # batches of data with a batch_size count 
        for step in range(0, n_classes, batch_size):
            end = step + batch_size
            batch_x, batch_y = x_train[step:end], y_train[step:end]
            _, train_accuracy = sess.run([training, accuracy], feed_dict={x: batch_x, y: batch_y})
        
        # check the accuracy of the model against the validation set
        validation_accuracy = sess.run(accuracy, feed_dict={x: mnist.validation.images, y:one_hot_valid})
        # print out the models accuracies. to print on same line, add \r to start of string
        sys.stdout.write("EPOCH {}. Train Accuracy = {:.3f},  Validation Accuracy = {:.3f}\n".format(epoch+1, train_accuracy, validation_accuracy))
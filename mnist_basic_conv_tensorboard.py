# -*- coding: utf-8 -*-
"""
**********************************************************************
* Filename    : mnist_basic_con_tensorboard.py
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

# Convolution layers require a height, width and depth
# reshape the input data from size 784 to (28 , 28) 
x_train_reshape = [data.reshape(28,28, 1) for data in mnist.train.images]
x_valid_reshape = [data.reshape(28,28, 1) for data in mnist.validation.images]

# display some random sample images to verify the input images are as we expect
random_num = np.random.randint(0, n_train)
# get a random image and reshape the data to a 28 x 28 tensor
image = mnist.train.images[random_num]
image = image.reshape((28,28))

plt.title("Number %d" % (mnist.train.labels[random_num]))
plt.axis('off')
plt.imshow(image, cmap=cm.Greys)

# Convolution layer
# each convolution layer will max pool with kernal size (2,2) and strides (2,2)
# this will result in halving the convolutioal layer
def convolution(x, filter_size, ksize=(3,3), strides=(1,1), \
                padding='SAME', name='conv'):
    # x_data: Input features from the precedding layer
    # filter_size: number of filter output from the convolution
    # ksize: kernal size 2D Tuple with kernel width and height
    # strides: kernel stride movement. 2D Tuple of width and height
    # padding: convolution padding type
    # return: a convolution tensor of the x_data
    with tf.name_scope(name):
        mean = 0
        sigma = 0.1
        
        input_features = x.get_shape().as_list()[-1] # number of features 
        
        # define the kernel shap for the convolutionaa filter
        shape = [ksize[0], ksize[1], input_features, filter_size]
        w = tf.Variable(tf.truncated_normal(shape=shape, mean=mean, stddev=sigma), 
                        name='W')
        b = tf.Variable(tf.zeros(filter_size), name='B')
        
        # perform a convolution layer on the x_data using strides conv_strides
        stride = [1, strides[0], strides[1], 1]
        conv = tf.nn.conv2d(x, w, strides=stride, padding=padding)
        conv = tf.nn.bias_add(conv, b)

        # activate the convolutional data
        relu = tf.nn.relu(conv)
        
        # return the max pool of the activation
        ksize = [1, 2, 2, 1]
        pool_strides = [1, 2, 2, 1]
        return tf.nn.max_pool(relu, ksize, pool_strides, padding='SAME')


# define a set structure for each linear transform layer
# also used on the final output layer
def linear_layer(x_data, node_count, name="linear_layer"):
    # x_data: Input features from the precedding layer
    # node_count: number of nodes outputing to in the next layer
    # return: activated output of current layer
    with tf.name_scope(name):
        input_features = x_data.get_shape().as_list()[1] # number of features 
    
        w = tf.Variable(tf.truncated_normal(shape=(input_features, node_count), \
                                            mean=0, stddev=0.1), name='W')
        b = tf.Variable(tf.zeros(node_count), name='B')
        
        return tf.add(tf.matmul(x_data, w), b)

# define a set structure for each hidden layer
def dense_layer(x_data, node_count, name="dense"):
    # x_data: Input features from the precedding layer
    # node_count: number of nodes outputing to in the next layer
    # return: activated output of current layer
    with tf.name_scope(name):
        linear = linear_layer(x_data, node_count, name=name)
        # activate the layer
        relu = tf.nn.relu(linear)
        return relu

# create the neural model for learning the data
# the model used is a two layer convolution with a two layer fully connected
def model(x, name='model'):
    # layer 1: convolution on data x with a filter count of 6
    # x : input data
    # name : tensorboard name
    # return : the final prediction
    with tf.name_scope(name):
        # Input: 28x28x3, Output: 14x14x6
        conv1 = convolution(x, 6)
        # Input: 14x14x6, Output: 7x7x16
        conv2 = convolution(conv1, 16)

        # flatten the convolutional layer before using in fully connected layer
        fc = tf.contrib.layers.flatten(conv2)

        # Layer 1: Input features = 784, Output = 100
        fc1 = dense_layer(fc, 100)
        # Layer 2: Input features = 100, Output = 10
        model_predict = linear_layer(fc1, n_classes)

        return model_predict

# creates a one hot encoded vectors for the training data
def one_hot_encode(labels):
    # labels: label position data to be one hot encoded, [3, ...]
    # return a tensor with one hot encoding of data of labels, 
    # eg [[0,0,0,1,0],[...]]
    depth = len(np.unique(labels))
    size = np.shape(labels)[0]
    one_hot = np.zeros((size, depth))
    one_hot[np.arange(size), labels] = 1
    return one_hot


# define the system placeholders
with tf.name_scope('placholders'):
    x = tf.placeholder(tf.float32, (None, 28, 28, 1), name='input_data')
    y = tf.placeholder(tf.int16, (None, n_classes), name='label_data')

# one hot encode the training, validation and test data
one_hot_train = one_hot_encode(mnist.train.labels)
one_hot_valid = one_hot_encode(mnist.validation.labels)
one_hot_test = one_hot_encode(mnist.test.labels)

# Hyper parameters
epochs = 100
learn_rate = 0.01
batch_size = 128

# run the model
logits = model(x)

with tf.name_scope('loss'):
    # get the final output of the model and find the loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y)
    loss = tf.reduce_mean(cross_entropy)

with tf.name_scope('train'):
    # Update the model to predict better results based of the training loss
    optimizer = tf.train.AdamOptimizer(learn_rate)
    # train the model to minimise the loss
    training = optimizer.minimize(loss)

with tf.name_scope('accuracy'):
    # Accuracy of the model measured against the validation set
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Merge all the summaries and write them to log_path location
merged = tf.summary.merge_all()

# train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # create tensorboard session at location log_path and save the graph there
    log_path = '/tmp/tensorboard/data/mnist'
    writer = tf.summary.FileWriter(log_path, graph=sess.graph)
    
    # Traing the model
    print("Training")
    for epoch in range(epochs):
        # shuffle the training data to prevent similar batches
        x_train, y_train = shuffle(x_train_reshape, one_hot_train)
        
        # train with ALL the training data per epoch, Training each pass with 
        # batches of data with a batch_size count 
        for step in range(0, n_classes, batch_size):
            end = step + batch_size
            batch_x, batch_y = x_train[step:end], y_train[step:end]
            _, train_accuracy = sess.run([training, accuracy], \
                                         feed_dict={x: batch_x, y: batch_y})
        
        # check the accuracy of the model against the validation set
        validation_accuracy = sess.run(accuracy, \
                                       feed_dict={x: x_valid_reshape, \
                                                  y:one_hot_valid})
        # print out the models accuracies. 
        # to print on same line, add \r to start of string
        sys.stdout.write("EPOCH {}. Train Accuracy = {:.3f},  Validation "\
                         "Accuracy = {:.3f}\n".format(epoch+1, train_accuracy,\
                                     validation_accuracy))

        print("Run the command line:\n" \
              "--> tensorboard --logdir={} " \
              "\nThen open http://0.0.0.0:6006/ into your web browser"
              .format(log_path))
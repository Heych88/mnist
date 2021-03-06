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
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorboard.plugins.beholder import Beholder
from tensorflow.python.saved_model import tag_constants
from sklearn.utils import shuffle
from random import randint
from matplotlib import pyplot as plt
import sys
import os

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


# directory locations to store tensorboard and saved data
log_path = '/tmp/tensorboard/data/mnist/'
save_path = './saved_model/'


def one_hot_encode(labels):
    '''
    Creates a one hot encoded vectors for the training data
    :param labels: Label position data to be one hot encoded, [3, ...]
    :return: A tensor with one hot encoding of data of labels, eg [[0,0,0,1,0],[...]]
    '''

    depth = len(np.unique(labels))
    size = np.shape(labels)[0]
    one_hot = np.zeros((size, depth))
    one_hot[np.arange(size), labels] = 1
    return one_hot

# one hot encode the training, validation and test data
one_hot_train = one_hot_encode(mnist.train.labels)
one_hot_valid = one_hot_encode(mnist.validation.labels)
one_hot_test = one_hot_encode(mnist.test.labels)

# Hyper parameters
epochs = 100
learn_rate = 0.01
batch_size = 128


def variable_summaries(var, name):
    '''
    Creates summaries of the passed in variable for TensorBoard visualization
    from TensorBoard: Visualizing Learning
    https://www.tensorflow.org/get_started/summaries_and_tensorboard
    :param var: Variable that will have its values saved for visualization
    :param name: A label name for the variable
    :return:
    '''

    with tf.name_scope('summaries_'+name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean_'+name, mean) # save as a tensorboard summary
        with tf.name_scope('stddev_'+name):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev_'+name, stddev)
        tf.summary.scalar('max_'+name, tf.reduce_max(var))
        tf.summary.scalar('min_'+name, tf.reduce_min(var))
        tf.summary.histogram('histogram_'+name, var)


def convolution(x, filter_size, ksize=(3,3), strides=(1,1), padding='SAME', name='conv'):
    '''
    Convolution layer where each layer will max pool with kernel size (2,2) and strides (2,2)
    this will result in halving the convolutioal layer
    :param x: Input features from the preceding layer
    :param filter_size: Number of filter output from the convolution
    :param ksize: Kernal size 2D Tuple with kernel width and height
    :param strides: Kernel stride movement. 2D Tuple of width and height
    :param padding: Convolution padding type
    :param name: A label name for the layer
    :return: A convolution tensor of the input data
    '''

    with tf.name_scope(name):
        mean = 0
        sigma = 0.1
        
        input_features = x.get_shape().as_list()[-1] # number of features 
        
        # define the kernel shap for the convolutionaa filter
        shape = [ksize[0], ksize[1], input_features, filter_size]
        weight = tf.Variable(tf.truncated_normal(shape=shape, mean=mean, stddev=sigma),
                             name='Weight')
        variable_summaries(weight, 'weight_'+name) # save the weight summary for tensorboard
        bias = tf.Variable(tf.zeros(filter_size), name='Bias')
        variable_summaries(bias, 'bias_'+name)  # save the bias value summary for tensorboard
        
        # perform a convolution layer on the x_data using strides conv_strides
        stride = [1, strides[0], strides[1], 1]
        conv = tf.nn.conv2d(x, weight, strides=stride, padding=padding)
        conv = tf.nn.bias_add(conv, bias)

        # activate the convolutional data
        relu = tf.nn.relu(conv)
        tf.summary.histogram('activation_' + name, relu)
        
        # return the max pool of the activation
        ksize = [1, 2, 2, 1]
        pool_strides = [1, 2, 2, 1]
        return tf.nn.max_pool(relu, ksize, pool_strides, padding='SAME')


def linear_layer(x_data, node_count, name="linear_layer"):
    '''
    Defines a set structure for each linear transform layer following the convolutional layers
    :param x_data: Input features from the preceding layer
    :param node_count: Number of nodes to out put to in the next layer
    :param name: A label name for the layer
    :return: Output tensor of the current layer
    '''

    with tf.name_scope(name):
        input_features = x_data.get_shape().as_list()[1] # number of features 
    
        weight = tf.Variable(tf.truncated_normal(shape=(input_features, node_count),
                                                 mean=0, stddev=0.1), name='Weight')
        variable_summaries(weight, 'weights_'+name)
        bias = tf.Variable(tf.zeros(node_count), name='Bias')
        variable_summaries(bias, 'bias_'+name)
        
        return tf.add(tf.matmul(x_data, weight), bias)


def dense_layer(x_data, node_count, name="dense"):
    '''
    Defines a set structure for each hidden layer
    :param x_data: Input features from the preceding layer
    :param node_count: Number of nodes to out put to in the next layer
    :param name: A label name for the layer
    :return: Activated tensor for the layer
    '''

    with tf.name_scope(name):
        linear = linear_layer(x_data, node_count, name=name)
        # activate the layer
        relu = tf.nn.relu(linear)
        tf.summary.histogram('activation_'+name, relu)
        return relu


def model(x, name='model'):
    '''
    Create the neural model for learning the data. The model used is a two layer
    convolution with a two layer fully connected output
    :param x: Input features from the preceding layer
    :param name: A label name for the model
    :return: Tensor of n_classes containing the model predictions
    '''

    with tf.name_scope(name):
        # Input: 28x28x3, Output: 14x14x6
        conv1 = convolution(x, 6, name='conv1')
        # Input: 14x14x6, Output: 7x7x16
        conv2 = convolution(conv1, 16, name='conv2')

        # flatten the convolutional layer before using in fully connected layer
        fc = tf.contrib.layers.flatten(conv2)

        # Layer 1: Input features = 784, Output = 100
        fc1 = dense_layer(fc, 100, name='dense1')
        # Layer 2: Input features = 100, Output = 10
        model_predict = linear_layer(fc1, n_classes, name='output')

        return model_predict


def Placeholders():
    '''
    Input and label placeholders for the model
    :return: Model input, model label
    '''

    with tf.name_scope('placholders'):
        x = tf.placeholder(tf.float32, (None, 28, 28, 1), name='input_data')
        y = tf.placeholder(tf.int16, (None, n_classes), name='label_data')

    with tf.name_scope('input_data_images'):
        # display some random sample images to verify the data is as we expect
        number_of_images = 12
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, number_of_images)

    return x, y


def create_optimizer(x, y):
    '''
    Creates the optimiser functions for training the network
    :param x: Model input placeholder
    :param y: Model label placeholder
    :return: Network output, Tensorboard tensor, optimiser function, accuracy tensor
    '''

    with tf.name_scope('model_logits'):
        # run the model
        logits = model(x)
        # Name logits Tensor, so that is can be loaded from disk after training
        logits = tf.identity(logits, name='logits')

    with tf.name_scope('loss'):
        # get the final output of the model and find the loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y)
        loss = tf.reduce_mean(cross_entropy)
        #tf.summary.scalar('cross_entropy', cross_entropy)
        tf.summary.scalar('mean_loss', loss)

    with tf.name_scope('train'):
        # Update the model to predict better results based of the training loss
        optimizer = tf.train.AdamOptimizer(learn_rate)
        # train the model to minimise the loss
        training = optimizer.minimize(loss, name='training')

    with tf.name_scope('accuracy'):
        # Accuracy of the model measured against the validation set
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them to log_path location
    merged_summary = tf.identity(tf.summary.merge_all(), name='merged_summary')

    return logits, training, accuracy, merged_summary


def train_model(sess, x, y, merged_summary, training, accuracy):
    '''
    Function that trains the the model.
    :param sess: Current tensorflow session
    :param x: Model input placeholder
    :param y: Model label placeholder
    :param merged_summary: Tensorboard summary tensor
    :param training: Optimiser function
    :param accuracy: Accuracy tensor
    :return:
    '''

    # create tensorboard session at location log_path and save the graph there
    writer = tf.summary.FileWriter(log_path, graph=sess.graph)
    beholder = Beholder(log_path)

    saver = tf.train.Saver()

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
            summary, _, train_accuracy = sess.run([merged_summary, training, accuracy],
                                         feed_dict={x: batch_x, y: batch_y})

        # add summaries to tensorboard
        writer.add_summary(summary, epoch)
        saver.save(sess, log_path + 'model.ckpt', epoch)

        beholder.update(session=sess)

        # check the accuracy of the model against the validation set
        validation_accuracy = sess.run(accuracy, feed_dict={x: x_valid_reshape, y:one_hot_valid})

        # print out the models accuracies.
        # to print on the same line, add \r to start of string
        sys.stdout.write("EPOCH {}. Train Accuracy = {:.3f},  Validation "
                         "Accuracy = {:.3f}\n".format(epoch+1, train_accuracy,
                                     validation_accuracy))

    saver_path = saver.save(sess, save_path+"model.ckpt")
    print("Model saved in path: %s" % saver_path)

    print("\nFinished")
    print("Run the command line:\n" \
          "--> tensorboard --logdir={} " \
          "\nThen open http://0.0.0.0:6006/ into your web browser"\
          .format(log_path))

    writer.close()



def load_last_model(sess):
    '''
    Loads the last checkpoint data for continual training or predictions
    :param sess: Current tensorflow session
    :return: Model input, model label, tensorboard tensor, optimiser function, accuracy tensor
    '''

    # Load meta graph and restore weights
    saver = tf.train.import_meta_graph(save_path + 'model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(save_path))

    # view all the graph tensor names
    #for op in sess.graph.get_operations():
    #    print(str(op.name))

    # Get Tensors from loaded model
    graph = tf.get_default_graph()

    loaded_x = graph.get_tensor_by_name('placholders/input_data:0')
    loaded_y = graph.get_tensor_by_name('placholders/label_data:0')
    loaded_merged_summary = graph.get_tensor_by_name('Merge/MergeSummary:0')
    loaded_accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')
    loaded_logits = graph.get_tensor_by_name('model_logits/logits:0')

    loaded_training = graph.get_operation_by_name("train/training")

    return loaded_x, loaded_y, loaded_merged_summary, loaded_logits, loaded_training, loaded_accuracy


def predict(logits):
    '''
    Predicts the class of the input image of the neural network.
    Prints the class and shows the input image.
    :param logits: Network model output.
    :return:
    '''
    num = randint(0, mnist.test.images.shape[0])
    img = mnist.test.images[num]

    prediction = tf.argmax(logits, 1)
    best = prediction.eval(feed_dict={x: img.reshape((1, 28, 28, 1))})
    print("Prediction ", best[0])

    plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
    plt.show()


if __name__ == '__main__':

    if os.path.isfile(save_path + 'model.ckpt.meta'):
        # continue training from the last checkpoint
        with tf.Session() as sess:
            x, y, merged_summary, logits, training, accuracy = load_last_model(sess)
            train_model(sess, x, y, merged_summary, training, accuracy)

            predict(logits)

    else:
        # create a new model and start training
        x, y, = Placeholders()
        logits, training, accuracy, merged_summary = create_optimizer(x, y)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_model(sess, x, y, merged_summary, training, accuracy)

            predict(logits)

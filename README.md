# mnist
This Repository shows different tensorflow methods of building ontop of the mnist data-set.

Each file deals with the same mnist data-set, but adds extra features to the code to create more accurate model.
The codes can be run seperatly for experimenting and visulising what different models and layers on the same data-set can produce.

#### mnist_basic_two_layer.py
A simple two layer neural-net classifyer.

#### mnist_basic_convolution_layer.py
A two layer convolution network with 2 dense layers

#### mnist_basic_conv_tensorboard.py
Same model as "mnist_basic_convolution_layer.py" but visulizing the system with TensorBoard.

### Running a model
1.Run "<the choosen file>.py" in your favorite python IDE or in terminal or command prompt

If the model uses Tensorboard

2. Run "tensorboard --logdir=/tmp/tensorboard/data/mnist" in termal or command prompt

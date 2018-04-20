# mnist
This Repository shows different tensorflow methods of building ontop of the mnist data-set.

Each file deals with the same mnist data-set but adds extra features to the code to create a more accurate model.
The codes can be run separately for experimenting and visualising what different models and layers on the same data-set will produce.

## Installation and setup
### Prerequisites

1. [Python 3](https://www.python.org/).

2. Install the required python packages.

```bash
pip install numpy scipy sklearn --user
```

3. [Tensorflow](https://www.tensorflow.org/install/).

### Build Instructions

1. Clone the project repository
```bash
git clone https://github.com/Heych88/mnist.git
```

## Running Code
### mnist_basic_two_layer.py
A simple two layer neural-net classifyer.

```bash
cd <download directory location>/mnist
python3 mnist_basic_two_layer.py
```

### mnist_basic_convolution_layer.py
A two layer convolution network with 2 dense layers.

```bash
cd <download directory location>/mnist
python3 mnist_basic_convolution_layer.py
```

### mnist_basic_conv_tensorboard.py
The same model as "mnist_basic_convolution_layer.py" but visualising the system with TensorBoard. This code can also continue training from a previously saved checkpoint.

```bash
cd <download directory location>/mnist
python3 mnist_basic_conv_tensorboard.py
```

Visualise the model uses Tensorboard. Open a new terminal or command prompt.

```bash
tensorboard --logdir=/tmp/tensorboard/data/mnist
```

Open a web browser and paste in the address bar http://0.0.0.0:6006/.

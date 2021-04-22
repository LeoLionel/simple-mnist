# A feed forward Neural Network for MNIST with NumPy

This repository contains the implementation of a simple feed forward neural network for handwritten digit classification on the MNIST dataset.

The code is based on the example in Michael Nielsens [(online) book](http://neuralnetworksanddeeplearning.com/) “Neural Networks and Deep Learning”. Additional features are: vectorization over mini batches, Adam optimizer, dropout, different activation and loss functions.

For training download the data (four files) from [here](http://yann.lecun.com/exdb/mnist/) and put them in a new subfolder `data/`, do not unzip them. Now you can train the network with the following code:


```
import load_mnist
import simple_nn

training_data, test_data = load_mnist.load_data()

net = simple_nn.Network([784,100,10], 'relu')
net.SGD_adam(training_data, test_data,
             epochs = 10,
             mini_batch_size = 10,
             alpha = 0.001, # learning rate
             dropout_rate = 0.1,
             lmbda = 0, # L2 regularization parameter

```

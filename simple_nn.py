import random
import time
import numpy as np


class Network(object):
    """ Create a feed-forward neural network.

    Example: Network([784, 30, 10], 'sigmoid') creates a network with
    input dimension 784, a hidden layer of dimension 30, and outout
    layer of dimension 10 with sigmoid activations.
    """

    def __init__(self, sizes: list, activation: str):
        """
        sizes: list of input, hidden and output dimensions
        activation: choice for the activation function
        """

        self.sizes = sizes
        self.initialize_weights_biases()

        if activation == 'sigmoid':
            self.activation              = sigmoid
            self.activation_deriv        = sigmoid_deriv
            self.activation_output_layer = sigmoid
            self.cost_fun                = cross_entropy_cost

        elif activation == 'relu':
            self.activation              = relu
            self.activation_deriv        = relu_deriv
            self.activation_output_layer = softmax
            self.cost_fun                = log_likelihood_cost

            self.weights = [w * 0.001 for w in self.weights]
            self.biases  = [b * 0.001 for b in self.biases]

        else:
            raise ValueError("Unknown activation function: " + repr(activation) +
                             ", use 'sigmoid' or 'relu'.")

        self.accuracies_test = []
        self.accuracies_train = []


    def initialize_weights_biases(self):
        """Initialize the starting weights and biases for the Network."""

        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]


    def feedforward(self, a):
        """Calculate the prediction of the network for input 'a'."""

        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, a) + b
            a = self.activation(z)

        b, w = self.biases[-1], self.weights[-1]
        z = np.dot(w, a) + b
        a  = self.activation_output_layer(z)

        return a


    def SGD(self, training_data, test_data, epochs, mini_batch_size,
            eta, lmbda=0, dropout_rate=0, monitor_accuracy=True):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The 'training_data' is a list of tuples
        (x, y) representing the training inputs and labels. (See
        documentation in load_mist.py for details of the data format.)
        Eta is the learning rate, lmbda the parameter for L2 regularization.
        If monitor_accuracy == True the network will be evaluated
        against the test and training data after each epoch. Partial
        progress is printed out and saved.
        """
        n = len(training_data)

        for j in range(epochs):
            start_time = time.time()
            random.shuffle(training_data)
            mini_batches = [ training_data[k:k+mini_batch_size]
                             for k in range(0, n, mini_batch_size)]

            for batch in mini_batches:
                # X and Y are matrices where each column is a single
                # input x or label y. So
                # X.shape == (784, mini_batch_size)
                # Y.shape == (10 , mini_batch_size)
                X = np.column_stack( tuple(x for (x,_) in batch) )
                Y = np.column_stack( tuple(y for (_,y) in batch) )

                nabla_b, nabla_w = self.backprop(X, Y, dropout_rate,
                                                 mini_batch_size)

                self.weights = [(1 - eta * lmbda / n ) * w - eta * nw
                                for w, nw in zip(self.weights, nabla_w)]

                self.biases  = [b - eta * nb
                                for b, nb in zip(self.biases, nabla_b)]

            self.progress_and_monitoring(training_data, test_data, epochs,
                                         monitor_accuracy, start_time, j+1)


    def SGD_adam(self, training_data, test_data, epochs, mini_batch_size,
                 alpha, lmbda=0, dropout_rate=0, monitor_accuracy=True):
        """A variation of the SDG method using the Adam optimization
        algorihthm as desrcibed in arXiv:1412.6980v9. The learning
        rate is renamed to alpha, the other parameters are as in the
        SGD method.
        """
        n = len(training_data)

        # parameters for the algorithm as suggested in the original paper
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 10e-8
        print('Starting with training')

        ms_w = [np.zeros(w.shape) for w in self.weights]
        vs_w = [np.zeros(w.shape) for w in self.weights]
        ms_b = [np.zeros(b.shape) for b in self.biases]
        vs_b = [np.zeros(b.shape) for b in self.biases]

        for j in range(epochs):
            start_time = time.time()
            random.shuffle(training_data)
            mini_batches = [ training_data[k:k+mini_batch_size]
                             for k in range(0, n, mini_batch_size)]

            for batch in mini_batches:
                X = np.column_stack( tuple(x for (x,_) in batch) )
                Y = np.column_stack( tuple(y for (_,y) in batch) )

                nabla_b, nabla_w = self.backprop(X, Y, dropout_rate,
                                                 mini_batch_size)

                ms_w = [beta1 * mw + (1-beta1)*nw for mw, nw in zip(ms_w, nabla_w)]
                ms_b = [beta1 * mb + (1-beta1)*nb for mb, nb in zip(ms_b, nabla_b)]

                vs_w = [beta2 * vw + (1-beta2) * nw * nw
                         for vw, nw in zip(vs_w, nabla_w)]
                vs_b = [beta2 * vb + (1-beta2) * nb * nb
                         for vb, nb in zip(vs_b, nabla_b)]
                #bias correction for the first iterations of Adam not implemented

                self.weights = [
                    (1 - alpha * lmbda/n)*w - alpha * mw / (np.sqrt(vw) + epsilon)
                                for w, mw, vw in zip(self.weights, ms_w, vs_w)]

                self.biases  = [b - alpha * mb / (np.sqrt(vb) + epsilon)
                                for b, mb,vb in zip(self.biases, ms_b, vs_b)]

            self.progress_and_monitoring(training_data, test_data, epochs,
                                         monitor_accuracy, start_time, j+1)



    def backprop(self, X, Y, dropout_rate, mini_batch_size):
        """Returns a tuple (nabla_b, nabla_w) representing the
        gradient for the cost function. nabla_b and nabla_w are
        layer-by-layer lists of numpy arrays, similar to self.biases
        and self.weights.
        """
        # -- Forward pass --
        As = [] # list to store all the activations, layer by layer
        Zs = []  # list to store all the Z matrices, layer by layer
        A = X
        As.append(X)

        p = 1 - dropout_rate # keep probability

        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            Z = np.dot(w, A) + b # broadcasting takes care of bs dimensions
            A = self.activation(Z)
            if dropout_rate:
                mask = np.random.rand(*Z.shape) < p
                A *= mask / p  # rescale here, not at prediction time
            Zs.append(Z)
            As.append(A)

        w, b = self.weights[-1], self.biases[-1]
        Z = np.dot(w, A) + b
        A = np.apply_along_axis(self.activation_output_layer,0,Z)
        # apply_along_axis because softmax needs to be applied per column
        Zs.append(Z)
        As.append(A)

        # -- Backward pass --
        num_layers = len(self.sizes)
        nabla_w = [None] * (num_layers - 1)
        nabla_b = [None] * (num_layers - 1)
        m = mini_batch_size

        # For a sigmoid output layer with cross entropy or a softmax output layer
        # with log likelihood loss the first step of backprop is identical.
        Delta = (As[-1] - Y)

        nabla_b[-1] = (1/m) * np.sum(Delta, axis=1, keepdims=True) # !
        nabla_w[-1] = (1/m) * np.dot(Delta, As[-2].transpose())    # !
        # For nabla_w the sum of the dot product is over the training
        # examples in the mini_batch.
        # nabla_w[-1][j][k] = sum( Delta[j][:] * A[k][:] )

        # The loop is over the layers of the network, l = 1 means
        # the last layer, l = 2 is the second-last layer, and so on.
        for l in range(2, num_layers):
            Z = Zs[-l]
            Ad = self.activation_deriv(Z)
            Delta = np.dot(self.weights[-l+1].transpose(), Delta) * Ad
            nabla_b[-l] = (1/m) * np.sum(Delta, axis=1, keepdims=True) # !
            nabla_w[-l] = (1/m) * np.dot(Delta, As[-l-1].transpose())  # !

        return (nabla_b, nabla_w)


    def progress_and_monitoring(self, training_data, test_data, epochs,
                                monitor_accuracy, start_time, current_epoch):
        "Print epoch, time and accuracy information."
        n = len(training_data)
        n_test = len(test_data)

        def progress_str (time):
            k = int(np.log(epochs)/np.log(10)) + 1
            return "Epoch {0:{pad}d}/{1:{pad}d} ({2:.1f}s)".format(
                current_epoch, epochs, time-start_time, pad=k)

        if monitor_accuracy:
            accuracy_test  = self.evaluate(test_data) / n_test * 100
            self.accuracies_test.append(accuracy_test)
            # Vectorization over a batch is just implemented for the
            # training method not for prediction. For speed, the
            # accuracy calculation is just done on a part of the training data.
            random.shuffle(training_data)
            training_data_part = training_data[:n_test]
            accuracy_train = self.evaluate(training_data_part) / n_test * 100
            self.accuracies_train.append(accuracy_train)
            accuracy_str = ", train / test accuracy: {0:.1f}% / {1:.1f}%".format(
                accuracy_train, accuracy_test)
            t = time.time()
            print(progress_str(t) + accuracy_str, end='\n')

        else:
            t = time.time()
            print(progress_str(t), end='\r')

        if current_epoch == epochs:
            print("\n"+
                "Accuracy on training and test data: {0:.1f}% / {1:.1f}%".format(
                    self.evaluate(training_data) / n * 100,
                    self.evaluate(test_data) / n_test * 100)
            )


    def evaluate(self, eval_data):
        """Return the number of cases in the eval_data for which
        the neural network predicts the correct result.
        """
        t = data_type(eval_data)

        if t == 'training':
            results = [ (np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in eval_data ]
        elif t == 'test':
            results = [ (np.argmax(self.feedforward(x)), y)
                        for (x, y) in eval_data ]
        else:
            raise TypeError('Data to evaluate the networks accuracy' +
                            'has wrong format.')

        correct_predictions = sum( [int(x == y) for (x, y) in results] )

        return correct_predictions


    def evaluate_cost(self, eval_data):
        """Calculate the total cost of eval_data.
        Caution! Just works if data_type(eval_data) == 'training'.
        """
        costs = [self.cost_fun(self.feedforward(x), y)
                 for x, y in eval_data]
        return sum(costs)


def data_type(data_set):
    """ Training data and test data have the output labels stored in a
    different format. Either as a one-hot vector or as an
    interger. This function detects which format is used by its input data_set."""

    first_img, first_label = data_set[0]
    if type(first_label) == np.ndarray:
        return 'training'
    elif type(first_label) == np.ubyte:
        return 'test'
    else:
        return 'unknown'


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(z):
    return sigmoid(z) * (1-sigmoid(z))

def cross_entropy_cost(a,y):
    c = -y * np.log(a) - (1-y) * np.log(1-a)
    return np.sum(np.nan_to_num(c))
    # np.nan_to_num(np.log(0)*0) == 0.0



def relu (z):
    return (abs(z) + z) / 2

def relu_deriv (z):
    return (np.sign(z) + 1) / 2

def softmax(z):
    x = np.exp(z)
    s = np.sum(x)
    return x/s

def log_likelihood_cost(a,y):
    c = -y * np.log(a)
    return np.sum(np.nan_to_num(c))
    # np.nan_to_num(np.log(0)*0) == 0.0

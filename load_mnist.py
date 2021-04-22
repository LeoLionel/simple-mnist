import gzip
import numpy as np


def load_data():
    """Loads the mnist data from the original files avaliable at:
    http://yann.lecun.com/exdb/mnist/ and returns the pair
    training_data and test_data.

    traing_data is a list of 60,000 (x, y) pairs, where x, the image
    data, is a numpy array of shape (784,1) and y, the label, a numpy
    array of shape (10,1).

    test_data is a list of 10,000 (x, y) pairs, where x, the image
    data, is a numpy array of shape (784,1) and y, the label, an
    integer.
    """
    print('Reading data from original files')
    training_data = read_files(
        "./data/train-images-idx3-ubyte.gz",
        "./data/train-labels-idx1-ubyte.gz", 60000)
    test_data = read_files(
        "./data/t10k-images-idx3-ubyte.gz",
        "./data/t10k-labels-idx1-ubyte.gz", 10000)

    training_data = normalize(training_data)
    test_data = normalize(test_data)

    training_data = transform(training_data, 'training')
    test_data = transform(test_data , 'test')

    return training_data, test_data


def normalize(data_set):
    images, labels = data_set
    return (images/256, labels)


def transform(data, data_format):
    """ Transform a pair of arrays with images and labels to a list of
    pairs (image, label). Depending on data_format label is either a
    number or unit vector.
    """
    images, labels = data
    if data_format == 'training':
        inputs  = [x.reshape(784, 1) for x in images]
        results = [unit_vector(y) for y in labels]
        return list(zip(inputs, results))

    elif data_format == 'test':
        inputs  = [x.reshape(784, 1) for x in images]
        results = labels
        return list(zip(inputs, results))


def read_files(img_file, label_file, n):
    """Load image and label data from mnist files. Returns a pair of
    numpy arrays with dimensions images.shape == (n,784) and
    labels.shape == (n, )

    For loading, np.frombuffer is used instead of np.fromfile, since
    the latter does not work with a gzip file object. See:
    https://github.com/numpy/numpy/issues/10866
    """
    dt = np.ubyte
    with gzip.open(img_file, "rb") as f:
        f.read(16)
        images = np.frombuffer(f.read(784*n), dtype=dt)
        images = images.reshape(n, 784)
    with gzip.open(label_file, "rb") as f:
        f.read(8)
        labels = np.frombuffer(f.read(n), dtype=dt)
    return images, labels


def unit_vector(i):
    """Return a 10-dimensional unit vector with a 1.0 in the jth position
    and zeroes elsewhere. This is used to convert a digit (0...9) into
    a corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[i.astype(int)] = 1.0
    return e


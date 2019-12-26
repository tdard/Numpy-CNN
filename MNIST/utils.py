import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_process_data(n_elem=10):
    sample_submission = pd.read_csv("digit-recognizer/sample_submission.csv")
    # dataframes
    test = pd.read_csv("digit-recognizer/test.csv")
    train = pd.read_csv("digit-recognizer/train.csv")
    
    # train labels
    Y_train = train.pop('label')
    Y_train = pd.get_dummies(Y_train).to_numpy()
    Y_train = Y_train.T
    
    # train features
    X_train = train.to_numpy()
    X_train = X_train.reshape(-1, 28, 28, 1)
    
    # test features
    X_test = test.to_numpy()
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # Additional crop: restrain set to retrieve only a few examples
    X_train = X_train[0:n_elem, ...]
    Y_train = Y_train[..., 0:n_elem]
    X_test = X_test[0:n_elem, ...]

    return X_train, Y_train, X_test


def plot_random_image(X, n=1, Y=None):
    m = X.shape[0]
    np.random.seed(1)
    print("Press any key to get next image, and q to quit")
    for k in range(n):
        i = np.random.randint(0, m)
        if Y is not None:
            label = np.argmax(Y[:, i])
            print("This image is a {}".format(label))
        plt.imshow(X[i, ...].reshape(28, 28))
        plt.show()
        key = input()
        if key == 'q':
            break

def initialize_weights(weight_shape):
    """
    For dense layer
    Initializes weights for a densely connected layer, according to Xavier initialization
    :weight_shape: (#units, #prev_units)
    """
    np.random.seed(1)
    units, prev_units = weight_shape
    weights = np.random.randn(units, prev_units)*np.sqrt(2/prev_units)
    bias = np.zeros((units, 1))
    return weights, bias

def initialize_weightsv2(weight_shape):
    # uniform initialization
    # for dense layer
    np.random.seed(1)
    units, prev_units = weight_shape
    epsilon_init = np.sqrt(6/(units+prev_units))
    weights = np.random.rand(units, prev_units) * 2 * epsilon_init - epsilon_init
    bias = np.zeros((units, 1))
    return weights, bias

def initialize_weights_conv(filters, kernel_size : int, prev_layer_shape):
    """
    Initializes weights for convolutional layer with Xavier initialization
    :filters: number of convolution filters
    :kernel_size: size of convolution kernel
    :prev_layer_shape: (?, H_in, W_in, C_in).
    """
    np.random.seed(1)
    H_in, W_in, C_in = prev_layer_shape[1:]
    weights = np.zeros((filters, kernel_size, kernel_size, C_in))
    weights = np.random.randn(filters, kernel_size, kernel_size, C_in) * np.sqrt(2 / (H_in * W_in * C_in + H_in * W_in * filters))
    bias = np.zeros((filters, 1, 1, 1))
    return weights, bias

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values = (0,)) # No padding over examples nor over channels
    return X_pad

def get_pad_from_string(pad_str, kernel_size):
    if pad_str.lower() == 'valid': # no padding
        pad = 0 # o = (i - k + 1)/s
    elif pad_str.lower() == 'same': # also known as half padding. Useful for odd kernel_size
        pad = int(kernel_size/2) # o = i/s for odd kernel_size
    elif pad_str.lower() == 'full': # increases output size: o = (i + k - 1)/s
        pad = kernel_size-1
    else:
        raise ValueError('Unkown padding pattern: {}'.format(pad_str))
    return pad


def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp = np.exp(Z) # (n, m)
    exp_sum = np.sum(exp, axis=0, keepdims=True) # (1, m)
    return exp/exp_sum

def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    return A

def activate(Z, activation):
        if activation == 'linear' or activation is None:
            A = Z
        elif activation == 'softmax':
            A = softmax(Z)
        elif activation == 'relu':
            A = relu(Z)
        elif activation == 'sigmoid':
            A = sigmoid(Z)
        else:
            raise ValueError('The activation is not implemented within this layer')
        return A

def activation_derivative(Z, activation):
        if activation == "linear" or activation is None:
            d = 1
        elif activation == 'sigmoid':
            d = sigmoid(Z) * (1 - sigmoid(Z))
        elif activation == 'relu':
            d = np.where(Z <= 0, 0, 1)
        else:
            raise ValueError('The activation is not implemented within this layer')
        return d


# Useless here but for deconvolutions it can be useful
def space_elements(Z, space):
    """
    :Z: (m, H, W, C)
    :space: int
    Spaces the elements of Z by a gap of value space.
    Return a matrix of shape (m, H + (H-1)*space, W + (W-1)*space, C)
    """
    Z_spaced = np.copy(Z)
    # Add columns of zeros
    for k in range(Z_spaced.shape[2]-1):
        for s in range(space):
            Z_spaced = np.insert(Z_spaced, k*(1+space)+1, np.zeros((1, 1, Z_spaced.shape[1], 1)), axis=2)
    # Add rows of zeros
    for k in range(Z_spaced.shape[1]-1):
        for s in range(space):
            Z_spaced = np.insert(Z_spaced, k*(1+space)+1, np.zeros((1, 1, Z_spaced.shape[2], 1)), axis=1)
    return Z_spaced


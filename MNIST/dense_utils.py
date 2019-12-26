from utils import *
from cost_utils import *
import numpy as np

class DenseLayer:
    def __init__(self, prev_layer, units, activation, input_shape=None):
        """
        :units: integer
        :input_shape: (n_A_prev, m)
        """
        if prev_layer is not None:
            self.input_shape = prev_layer.shape
        elif input_shape is not None:
            self.input_shape = input_shape
        else:
            raise ValueError('Missing shape or previous layer')
        self.shape = (units, None)
        self.units = units
        self.weights, self.bias = initialize_weights((units, self.input_shape[0]))
        self.activation = activation
    
        
    def forward(self, X, caches=None):
        """
        :X: (n_X, m) matrix. n is the number of features and m the number of examples
        :caches: is a list of caches that we use in training. Each element of cache is (A, Z) or (A,) depending if the layer has an activation
        """
        Z = np.dot(self.weights, X) + self.bias
        A = activate(Z, self.activation)
        
        # Training:
        if caches is not None:
            caches.append((A, Z))
        return A
    
    def backward(self, dA, cache, cache_prev, dWdb_caches):
        """
        :dA: error term of the output activation in this layer (n_A, m)
        :cache: (A, Z)
        :cache_prev: (A_prev, Z_prev) or (A_prev)
        :dWdb_caches: list of gradients
        """
        Z = cache[1]
        A_prev = cache_prev[0]
        
        # Number of examples
        m = Z.shape[-1]
        
        # Derivatives of activation
        dZ = activation_derivative(Z, self.activation) * dA
        
        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(self.weights.T, dZ)
        
        # Update caches
        dWdb_caches.append((dW, db))
        
        return dA_prev
     
    def backward_last_layer(self, dZ, Y, cache, cache_prev, dWdb_caches):
        """
        :dZ: error term of the output activation in this layer (n_A, m)
        :Y: ground truth
        :cache: (A, Z)
        :cache_prev: (A_prev, Z_prev) or (A_prev)
        :dWdb_caches: list of gradients
        """
        Z = cache[1]
        A_prev = cache_prev[0]
        
        # Number of examples
        n, m = Z.shape
        
        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(self.weights.T, dZ)
        
        # Update caches
        dWdb_caches.append((dW, db))

        return dA_prev
            
            
    def update(self, dW, db, learning_rate=0.01):
        """
        :learning_rate: alpha
        """
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db


class FlattenLayer:
    def __init__(self, prev_layer, input_shape=None):
        if prev_layer is not None:
            self.input_shape = prev_layer.shape
        elif input_shape is not None:
            self.input_shape = input_shape
        else:
            raise ValueError('Missing shape or previous layer')
        self.shape = (np.prod(self.input_shape[1:]), None)
        
    def forward(self, X, caches=None):
        """
        :caches: is a list of caches that we use in training
        """
        m = X.shape[0]
        A = np.reshape(X, (m, -1))
        A = A.T # (n_A, m)
        
        if caches is not None:
            caches.append((A,))
            
        return A
        
    def backward(self, dA, cache, cache_prev, dWdb_caches):
        """
        :dA: (units, m)
        cache and cache_prev are not used here
        dWdb_caches gets an empty tuple added 
        """
        m = dA.shape[1]
        H, W, C = self.input_shape[1:]
        
        # Reshape
        dA_prev = dA.T # (m, units)
        dA_prev = np.reshape(dA, (-1, H, W, C))
        
        assert dA_prev.shape == (m, H, W, C)
        
        # Update dWdb_caches with an empty tuple
        dWdb_caches.append(())
        
        return dA_prev

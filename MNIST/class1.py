from utils import *


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
        :caches: is a list of caches that we use in training
        """
        Z = np.dot(self.weights, X) + self.bias
        A = activate(Z, self.activation)
        
        # Training:
        if caches is not None:
            caches.append((A, Z))
        return A
    
    def backward_last_layer(self, Ygt, cache, cache_prev, dWdb_caches):
        """
        :Ygt: ground truth label (n_Y, m)
        :cache: (A, Z)
        :cache_prev: (A_prev, Z_prev) or (A_prev)
        :dWdb_caches: list of gradients
        """
        A = cache[0]
        A_prev = cache_prev[0]
        
        # First dA
        dA = A - Ygt
        dZ = dA 
        
        # Number of examples
        m = Ygt.shape[-1]
        
        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(self.weights.T, dZ)
        
        # Update caches
        dWdb_caches.append((dW, db))
         
        return dA_prev
    
    
    
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
        
    
    def update(self, dW, db, learning_rate=0.01):
        """
        :learning_rate: alpha
        """
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db
        
from dense_utils import *
from conv_utils import *
from cost_utils import *
import numpy as np


class Model:
    def __init__(self, layers=None):
        self.layers = [] if layers is None else layers
        self.input_shape = None if self.layers == [] else self.layers[0].input_shape
        
    def set_layers(self, layers):
        self.layers = layers
        self.input_shape = layers[0].input_shape
    
    def predict(self, X):
        A = X
        for l in self.layers:
                A = l.forward(A)
        pred_classes = np.argmax(A, axis=0) # 1D array with class indices from 0 to N_classes-1
        return pred_classes
  
    def compute_cost(self, X, Y):
        L = len(self.layers)
        # Forward propagation
        A = X
        caches = [(A,)] # will collect the output of each layer L: either (A,) for flatten pool and input, either (A, Z) for conv and dense
        for layer in self.layers: 
            A = layer.forward(A, caches=caches)
        # Compute cost
        cost = categorical_cross_entropy(A, Y)    

        # Back propagation
        dWdb_caches = []
        dZ = dZ_categorical_cross_entropy(A, Y)
        cache = caches.pop() # A, Z
        cache_prev = caches.pop() # (A_prev, Z_prev) or (A_prev,)

        assert isinstance(self.layers[-1], DenseLayer)

        dA = self.layers[-1].backward_last_layer(dZ, Y, cache, cache_prev, dWdb_caches)
        for l, layer in enumerate(reversed(self.layers[:-1])):
            cache = cache_prev
            cache_prev = caches.pop()
            dA = layer.backward(dA, cache, cache_prev, dWdb_caches)
 
        dWdb_caches = dWdb_caches[::-1] # Reverse gradient caches
        
        return cost, dWdb_caches

    def compute_numerical_gradients(self, X, Y):
        """
        Numerical gradient computed
        dWij = (J(W11, ..., Wij + epsilon, ...) - J(W11, ..., Wij - epsilon, ...)/(2*epsilon)
        """
        epsilon = 0.0001
        grads = []
        for layer in self.layers:
            print(type(layer))
            if isinstance(layer, DenseLayer) or isinstance(layer, Conv2DLayer):
                # Weights
                shape = layer.weights.shape
                n = layer.weights.size
                print("There are {} gradients to compute in this layer".format(n))
                dW = np.zeros((n, 1))
                for i in range(n):
                    # J_minus
                    layer.weights = layer.weights.reshape(-1)
                    layer.weights[i] -= epsilon
                    layer.weights = layer.weights.reshape(shape)
                    J_minus, _ = self.compute_cost(X, Y)
                    # J_plus
                    layer.weights = layer.weights.reshape(-1)
                    layer.weights[i] += 2*epsilon
                    layer.weights = layer.weights.reshape(shape)
                    J_plus, _ = self.compute_cost(X, Y)
                    # Set back W
                    layer.weights = layer.weights.reshape(-1)
                    layer.weights[i] -= epsilon
                    layer.weights = layer.weights.reshape(shape)
                    # dW
                    dW[i] = (J_plus - J_minus)/(2*epsilon)
                    print("{}/{} gradient computed".format(i+1, n))
                dW = dW.reshape(shape)

                # Bias
                n = layer.bias.size
                db = np.zeros((n, 1))
                for i in range(n):
                    layer.bias[i] -= epsilon
                    J_minus, _ = self.compute_cost(X, Y)
                    layer.bias[i] += 2*epsilon
                    J_plus, _ = self.compute_cost(X, Y)
                    layer.bias[i] -= epsilon
                    db[i] = (J_plus - J_minus)/(2*epsilon)

                grads.append((dW, db))
            else:
                grads.append(())
        return grads   
    
    def check_gradients(self, X, Y):
        print("Compute gradients using back prop...")
        _, back_prop_grads = self.compute_cost(X, Y)
        print("Compute numerical gradients...")
        numerical_grads = self.compute_numerical_gradients(X, Y)
        print("If the backpropagation implementation is correct, the relative difference will be small")
        for bp_grad, num_grad in zip(back_prop_grads, numerical_grads):
            if not(bp_grad == () or num_grad == ()):
                bp_dW, bp_db = bp_grad
                num_dW, num_db = num_grad
                diff_dW = relative_difference(num_dW, bp_dW)
                diff_db = relative_difference(num_db, bp_db)
                print("Difference dW: {}".format(diff_dW))
                print("Difference db: {}".format(diff_db))

    def iterate_on_minibatch(self, X_batch, Y_batch, learning_rate=0.01):
        cost, gradients = self.compute_cost(X_batch, Y_batch)
        for l in range(len(self.layers)):
            if len(gradients[l]) != 0:
                assert (isinstance(self.layers[l], DenseLayer) or isinstance(self.layers[l], Conv2DLayer))

                dW, db = gradients[l]
                self.layers[l].update(dW, db, learning_rate)
        return cost

    def train_on_batch(self, X, Y, learning_rate = 0.01, epochs=15):
        print("Batch gradient descent, without optimization")
        print("Learning rate: {}".format(learning_rate))
        costs = np.zeros((epochs, ))
        for e in range(1, epochs+1):
            cost = self.iterate_on_minibatch(X, Y, learning_rate)
            costs[e-1] = cost
            print("The cost for the epoch {} is: {}".format(e, cost))
        return costs




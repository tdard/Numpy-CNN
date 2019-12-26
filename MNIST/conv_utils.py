import numpy as np
from utils import *

def convolve_on_indices(I, K, bias, indices : tuple):
    """
    :I: (H, W, C) input array
    :K: (k, k, C) kernel array
    :bias: (1, 1, 1) bias array
    indices i,j
    The convolution operation of an image I with a kernel K is equivalent to the cross-correlation of the image with a flipped (180d) kernel
    """
    k = K.shape[0]
    i, j = indices
    # Kernel flip
    K_flipped_180 = np.rot90(K, 2, axes=(0, 1))
    # Cross correlation
    cross_cor = np.sum(K_flipped_180 * I[i:i+k, j:j+k])
    # Add bias
    res = cross_cor + float(bias)
    
    return res


def convolve_whole_image(I, K, bias, strides, pad_mode):
    """
    :I: (m, H_in, W_in, C) input array
    :K: (k, k, C) kernel array
    :bias: (1, 1, 1) bias array
    :strides: int
    :pad_mode: str

    Convolve a (m, H_in, W_in, C) image with one filter kernel K (k, k, C)
    Returns a (m, H_out, W_out) output
    """
    k = K.shape[0]
    m, H_in, W_in, C = I.shape
    pad = get_pad_from_string(pad_mode, k)

    H_out, W_out = int((H_in - k + 2*pad)/strides) + 1,  int((W_in - k + 2*pad)/strides)+1

    # Pad input image accordingly to padding mode
    I_pad = zero_pad(I, pad)
    # Initialize output image
    Z = np.zeros((m, H_out, W_out))
    # Loop over m, H_out and W_out
    for i in range(m):
        for h in range(H_out):
            h_start = h * strides
            for w in range(W_out):
                w_start = w * strides
                Z[i, h, w] = convolve_on_indices(I_pad[i, ...], 
                                                 K, 
                                                 bias, 
                                                 (h_start, w_start))
    return Z


class Conv2DLayer:
    def __init__(self, prev_layer, filters : int, kernel_size : int, strides : int, pad='same', activation='relu', input_shape=None):
        if prev_layer is not None:
            self.input_shape = prev_layer.shape
        elif input_shape is not None:
            self.input_shape = input_shape
        else:
            raise ValueError('Missing shape or previous layer')
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pad = pad

        H_in, W_in, C_in = self.input_shape[1:]
        pad = get_pad_from_string(pad, kernel_size)
        H_out = int((H_in - kernel_size + 2*pad)/strides)+1
        W_out = int((W_in - kernel_size + 2*pad)/strides)+1 
        self.shape = (None, H_out, W_out, filters)
        
        self.activation = activation
        self.weights, self.bias  = initialize_weights_conv(filters, kernel_size, self.input_shape) 
    
    def forward(self, X, caches=None):
        m, H_in, W_in = X.shape[:-1]
        H_out, W_out = self.shape[1:3]
        
        Z = np.zeros((m, H_out, W_out, self.filters))
        
        pad = get_pad_from_string(self.pad, self.kernel_size)
        X_pad = zero_pad(X, pad)
        # "Cross-correlation like CNN implementation"
        for i in range(m):
            for f in range(self.filters):
                for h in range(H_out):
                    for w in range(W_out):
                        Z[i, h, w, f] += np.sum(X_pad[i, 
                                                      h*self.strides:h*self.strides+self.kernel_size, 
                                                      w*self.strides:w*self.strides+self.kernel_size, 
                                                      :] * self.weights[f, ...]) + float(self.bias[f, ...])
        A = activate(Z, self.activation)

        assert A.shape == (m, H_out, W_out, self.filters)

        if caches is not None:
            caches.append((A, Z))

        return A

    def forwardv2(self, X, caches=None):
        """
        :X: is a (m, H_in, W_in, C_in) input
        :caches: is a list of caches that we use in training
        """
        m, H_in, W_in = X.shape[:-1]
        H_out, W_out = self.shape[1:3]
        Z = np.zeros((m, H_out, W_out, self.filters))
        
        for f in range(self.filters):
            # convolve_whole_image convolves I with shape (m, H_in, W_in, C_in) with a kernel filter K of shape (k, k, C_in)
            Z[..., f] = convolve_whole_image(X, self.weights[f, ...], self.bias[f,...], self.strides, self.pad)
        
        A = activate(Z, self.activation)
        
        assert A.shape == (m, H_out, W_out, self.filters)

        if caches is not None:
            caches.append((A, Z))

        return A
    
    def backward(self, dA, cache, cache_prev, dWdb_caches):
        """
        :dZ: gradient of the cost with respect to the output of this layer (m, H_out, W_out, filters)
        :A_prev: output of the previous layer
        """        
        Z = cache[1]
        A_prev = cache_prev[0]
        A_prev_padded = zero_pad(A_prev, get_pad_from_string(self.pad, self.kernel_size)) # Need it to compute the gradient dW

        H_in, W_in, C_in = A_prev.shape[1:]
        m, H_out, W_out = dA.shape[:-1]
        
        # Derivatives of activation
        dZ = activation_derivative(Z, self.activation) * dA

        # Compute db
        db = np.zeros((self.filters, 1, 1, 1))
        for f in range(self.filters):
            db[f, ...] = np.sum(dZ[..., f])
        db = 1/m * db

        # Compute dW
        # Transpose convolution of a dZ with kernel A_prev
        dW = np.zeros((self.weights.shape))
        for i in range(m):
            for h in range(self.kernel_size):
                for w in range(self.kernel_size):
                    for f in range(self.filters):
                        dW[f, h, w, :] += self.compute_dW_fij(A_prev_m=A_prev_padded[i, ...],  # (H_in, W_in, C_in)
                                                              dZ_mf=dZ[i, :, :, f], # (H_out, W_out)
                                                              indices=(h, w))
        dW = 1/m * dW

        # Compute dA_prev
        dA_prev = np.zeros(A_prev.shape)
        for i in range(m):
            for h in range(H_in):
                for w in range(W_in):
                    for c in range(C_in):
                        dA_prev[i, h, w, c] = self.compute_dAijc(W=self.weights, 
                                                                 dZ_m=dZ[i, ...], 
                                                                 indices=(h, w, c))
        
        dWdb_caches.append((dW, db))

        assert dA_prev.shape == (m, H_in, W_in, C_in)

        return dA_prev

    def backwardv2(self, dA, cache, cache_prev, dWdb_caches):
        """
        :dZ: gradient of the cost with respect to the output of this layer (m, H_out, W_out, filters)
        :A_prev: output of the previous layer
        """
        m, H_out, W_out = dA.shape[:-1]
        
        Z = cache[1]
        A_prev = cache_prev[0]
        
        H_in, W_in, C_in = A_prev.shape[1:]
        
        # Derivatives of activation
        dZ = dA * activation_derivative(Z, self.activation)
        
        dA_prev = np.zeros((m, H_in, W_in, C_in))
        dW = np.zeros((self.filters, self.kernel_size, self.kernel_size, C_in))
        db = np.zeros((self.filters, 1, 1, 1))
        
        pad = get_pad_from_string(self.pad, self.kernel_size)
        
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)

        for i in range(m):
            a_prev_pad = A_prev_pad[i, ...]
            da_prev_pad = dA_prev_pad[i, ...]
            for h in range(H_out):
                vert_start = self.strides * h
                vert_end = vert_start + self.kernel_size
                for w in range(W_out):
                    horiz_start = self.strides * w
                    horiz_end = horiz_start + self.kernel_size
                    for f in range(self.filters):
                        a_slice = a_prev_pad[vert_start : vert_end, horiz_start : horiz_end, :]
                        da_prev_pad[vert_start : vert_end, horiz_start : horiz_end, :] += self.weights[f, ...] * dZ[i, h, w, f]
                        dW[f, :, :, :] += a_slice * dZ[i, h, w, f] 
                        db[f, :, :, :] += dZ[i, h, w, f] 
            # Remove padding
            if pad == 0:
                dA_prev[i, ...] = da_prev_pad
            else:
                dA_prev[i, ...] = da_prev_pad[pad : -pad, pad : -pad, :]
            
            # Update caches
            dWdb_caches.append((dW, db))
            
            assert dA_prev.shape == (m, H_in, W_in, C_in)
            
            return dA_prev

    def compute_dW_fij(self, A_prev_m, dZ_mf, indices : tuple):
        """
        :A_prev_m: (H_in, W_in, C) input array
        :dZ_mf: (H_out, W_out) kernel array
        indices i,j
        Computes the cross correlation of dZij with A_prev_m
        Outputs a (C,) array representing dW_fijc for all c
        """    
        H_in, W_in, C = A_prev_m.shape
        H_out, W_out = dZ_mf.shape
        i, j = indices

        assert H_out == W_out

        dW_fij = np.zeros((C,))
        for i_prime in range(H_out):
            if i + i_prime >= H_in:
                break
            for j_prime in range(W_out):
                if j + j_prime >= W_in:
                    break
                dW_fij[:] += dZ_mf[i_prime, j_prime] * A_prev_m[i + i_prime, j + j_prime, :]

        return dW_fij

    def compute_dAijc(self, W, dZ_m, indices):
        """
        :W: (F, k, k, C) input array
        :dZ_m: (H_out, W_out, F) kernel array
        indices i, j, c
        """
        i, j, c = indices
        H_out, W_out, F = dZ_m.shape
        k = W.shape[1]

        assert W.all() == self.weights.all()
        assert H_out == W_out
        
        dAijc = 0
        
        for i_prime in range(H_out):
            for j_prime in range(W_out):
                for f in range(F):
                    if not(i - i_prime < 0 or i >= k or j - j_prime < 0 or j >= k):
                        dAijc += dZ_m[i_prime, j_prime, f] * W[f, i - i_prime, j - j_prime, c]
        return dAijc  

    def update(self, dW, db, learning_rate=0.01):
        """
        :learning_rate: alpha
        :dW:
        :db:
        """
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db


class Pooling2DLayer:
    def __init__(self, prev_layer, pool_size : int, strides : int, mode='max', input_shape=None):
        if prev_layer is not None:
            self.input_shape = prev_layer.shape
        elif input_shape is not None:
            self.input_shape = input_shape
        else:
            raise ValueError('Missing shape or previous layer')
        
        
        self.pool_size = pool_size
        self.strides = strides
        self.mode = mode
        
        H_in, W_in, C_in = self.input_shape[1:]
        H_out = int((H_in - pool_size)/strides) + 1
        W_out = int((W_in - pool_size)/strides) + 1
        C_out = C_in
        self.shape = (None, H_out, W_out, C_out)
    
    def forward(self, X, caches=None):
        """
        :X: is a (m, H_in, W_in, C_in) input
        :caches: is a list of caches that we use in training
        """
        m, H_in, W_in, C_in = X.shape
        H_out, W_out, C_out = self.shape[1:]
        
        A = np.zeros((m, H_out, W_out, C_out))
        for i in range(m):
            for h in range(H_out):
                vert_start = self.strides * h
                vert_end = vert_start + self.pool_size
                for w in range(W_out):
                    horiz_start = self.strides * w
                    horiz_end = horiz_start + self.pool_size
                    for c in range(C_out):
                        x_slice = X[i, vert_start : vert_end, horiz_start : horiz_end, c]
                        if self.mode == 'max':
                            A[i, h, w, c] = np.max(x_slice)
                        elif self.mode == 'avg' or self.mode == 'average':
                            A[i, h, w, c] = np.mean(x_slice)
        
        assert A.shape == (m, H_out, W_out, C_out)

        if caches is not None:
            caches.append((A,))
        
        return A
    
    def backward(self, dA, cache, cache_prev, dWdb_caches):
        """
        :dA: gradient of cost with respect to the output of the pooling layer, same shape as the output of the pooling layer. It is equivalent to dZ if no activation.
        :cache: (A, None). Not used here.
        :cache_prev: (A_prev, Z_prev) or (A_prev)
        :dWdb_caches: [(Wi, bi), (), ]
        """
        A_prev = cache_prev[0]
        
        m, H_in, W_in = A_prev.shape[:-1]
        H_out, W_out, C_out = dA.shape[1:]
        
        dA_prev = np.zeros((m, H_in, W_in, C_out))
        
        for i in range(m):
            a_prev = A_prev[i, ...]
            for h in range(H_out):
                vert_start = self.strides * h
                vert_end = vert_start + self.pool_size
                for w in range(W_out):
                    horiz_start = self.strides * w
                    horiz_end = horiz_start + self.pool_size
                    for c in range(C_out):
                        if self.mode == 'max':
                            a_prev_slice = a_prev[vert_start : vert_end, horiz_start : horiz_end, c]
                            mask = a_prev_slice == np.max(a_prev_slice) #  boolean mask on the maximum value of a_prev_slice
                            dA_prev[i, vert_start : vert_end, horiz_start : horiz_end, c] += mask * dA[i, h, w, c]
                        elif self.mode == 'average' or self.mode == 'avg':
                            da = dA[i, h, w, c]
                            dA_prev[i, vert_start : vert_end, horiz_start : horiz_end, c] += da/(self.pool_size*self.pool_size) * np.ones((self.pool_size, self.pool_size)) 
         
        assert dA_prev.shape == A_prev.shape

        # Update dWdb_caches with an empty tuple
        dWdb_caches.append(())
        
        return dA_prev
                        
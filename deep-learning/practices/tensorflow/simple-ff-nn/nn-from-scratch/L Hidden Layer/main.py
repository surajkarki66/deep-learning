import tensorflow as tf
import numpy as np

class Main:
    def sigmoid_forward(self, z):
        s = tf.nn.sigmoid(z)
        cache = z
        return s, cache

    
    def sigmoid_backward(self, da, cache):
        z = cache
        s, cache = self.sigmoid_forward(z)
        dz = da * s 
        dz = tf.multiply(dz, 1 - s)

        return dz

    def relu_forward(self, z):
        a = tf.maximum(0, z)
        cache = z

        return a, cache


    def relu_backward(self, da, cache):
        z = cache
        dz = np.array(da, copy=True)
        z = np.array(z, copy=True)
        # When z <= 0, you should set dz to 0 as well.
        dz[z <= 0] = 0    

        return dz

        
    def initialize_parameters(self, layer_dims):
        xavier = tf.initializers.GlorotUniform()

        L = len(layer_dims)
        parameters = {}

        for l in range(1, L):
            parameters['W' + str(l)] = tf.Variable(xavier(shape=(layer_dims[l], layer_dims[l-1])))
            parameters['b' + str(l)] = tf.zeros([layer_dims[l],1], dtype=tf.float32)

        return parameters

    
    def linear_forward(self, A, W, b):
        Z = tf.matmul(W, A) + b
       
        cache = ( A, W, b)

        return Z, cache


    def linear_activation_forward(self, A_prev, W, b, activation):

        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid_forward(Z)

        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu_forward(Z)

        cache = (linear_cache, activation_cache)

        return A, cache


    def forward_propagation(self, X, parameters):
        caches = []
        A = X
        L = len(parameters) // 2     
        
        # LINEAR -> RELU
        for l in range(1, L):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
            caches.append(cache)
        
        # LINEAR -> SIGMOID
        AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
        caches.append(cache)
                
        return AL, caches


    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = (-1/m) * (tf.matmul(Y, tf.transpose(tf.math.log(AL))) + tf.matmul(1-Y, tf.transpose(tf.math.log(1-AL))))            
        return cost

    
    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1. / m) * tf.matmul(dZ, tf.transpose(cache[0])) 
        db = (1. / m) * tf.reduce_sum(dZ, axis=1, keepdims=True)
        dA_prev = tf.matmul(tf.transpose(cache[1]), dZ)

        return dA_prev, dW, db
    

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            
        return dA_prev, dW, db

    
    def backward_propagation(self, AL, Y, caches, parameters):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = np.array(Y)
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        Y = tf.cast(Y, tf.float32)
        # Initializing the backpropagation
        dAL = - (tf.divide(Y, AL) - tf.divide(1 - Y, 1 - AL))
        
        # lth layer: (Sigmoid -> Linear) gradients
        current_cache = caches[L-1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activation = "sigmoid")
        
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads


    
    def update_parameters(self, parameters, grads, learning_rate):
        
        L = len(parameters) // 2 

        # updation
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
            
        return parameters
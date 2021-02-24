import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import tensorflow as tf

from utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets, preprocessing


def initialize_parameters(layer_dims):
    xavier = tf.initializers.GlorotUniform()

    L = len(layer_dims)
    parameters = {}

    for l in range(1, L):
        parameters['W' + str(l)] = tf.Variable(xavier(shape=(layer_dims[l], layer_dims[l-1])))
        parameters['W' + str(l)] = tf.cast(parameters['W' + str(l)] , tf.float64)

        parameters['b' + str(l)] = tf.zeros([layer_dims[l],1], dtype=tf.float64)

    return parameters


def forward_propagation(X, parameters):
    # retrieve parameters
    L = len(parameters) // 2
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    W4 = parameters["W4"]
    b4 = parameters["b4"]

    z1 = tf.matmul(W1, X) + b1
    a1 = tf.nn.relu(z1)
    z2 = tf.matmul(W2, a1) + b2
    a2 = tf.nn.relu(z2)
    z3 = tf.matmul(W3, a2) + b3
    a3 = tf.nn.relu(z3)
    z4 = tf.matmul(W4, a3) + b4
    a4 = sigmoid(z4)

    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3, z4, a4, W4, b4)

    return a4, cache




def compute_cost(A4, Y):
    m = Y.shape[1]
    cost = (-1/m) * (tf.matmul(Y, tf.transpose(tf.math.log(A4))) + tf.matmul(1-Y, tf.transpose(tf.math.log(1-A4))))
    cost = tf.cast(cost, tf.float64)
                  
    return cost


def backward_propagation(X, Y, cache):
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3, z4, a4, W4, b4)  = cache


    dz4 = 1./m * (a4 - Y)
    dW4 = tf.matmul(dz4, tf.transpose(a3))
    db4 = tf.reduce_sum(dz4, axis=1, keepdims = True)
    
    da3 = tf.matmul(tf.transpose(W4), dz4)
    dz3 = tf.multiply(da3, tf.cast(a3 > 0, tf.float64))
    dW3 = tf.matmul(dz3, tf.transpose(a2))
    db3 = tf.reduce_sum(dz3, axis=1, keepdims=True)

    da2 = tf.matmul(tf.transpose(W3), dz3)
    dz2 = tf.multiply(da2, tf.cast(a2 > 0, tf.float64))
    dW2 = tf.matmul(dz2, tf.transpose(a1))
    db2 = tf.reduce_sum(dz2, axis=1, keepdims=True)
    
    da1 = tf.matmul(tf.transpose(W2), dz2)
    dz1 = tf.multiply(da1, tf.cast(a1 > 0, tf.float64))
    dW1 = tf.matmul(dz1, tf.transpose(X))
    db1 = tf.reduce_sum(dz1, axis=1, keepdims = True)
    
    gradients = {"dz4": dz4, "dW4": dW4, "db4": db4,
                 "dz3": dz3, "dW3": dW3, "db3": db3, "da3": da3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}

    return gradients



def update_parameters(parameters, grads, learning_rate=0.0002):
    L = len(parameters) // 2
    for l in range(L):
        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - learning_rate * grads['dW' + str(l+1)]
        parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - learning_rate * grads['db' + str(l+1)]
    
    return parameters



def nn_model(X, Y, num_iterations=10000, print_cost = False, layer_dims=[2, 10, 5, 5, 1]):

    X = tf.cast(X, tf.float64)
    Y = tf.cast(Y, tf.float64)
    
    # Initialize parameters
    parameters = initialize_parameters(layer_dims)
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward Propagation
        A4, cache = forward_propagation(X, parameters)
        
        #cost function
        cost = compute_cost(A4, Y)
        
        # Backpropagation
        grads = backward_propagation(X, Y, cache)

        # parameter update
        parameters = update_parameters(parameters, grads, learning_rate=0.01)
        
        #print(parameters)
        if print_cost and i % 1000 == 0:
            print("Cost after iteration: %i: %f " %(i, cost))
            
    return parameters



def predict(parameters, X):
    X = tf.cast(X, tf.float64)

    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    predictions = np.array(predictions, dtype=np.bool)
    return predictions

    
    



if __name__ == "__main__": 

    #X, Y = load_planar_dataset()
    train_x, test_x, train_y, test_y, classes, num_px= preprocessing()
    layer_dims = [12288, 20, 20, 5, 1]
    parameters = nn_model(train_x, train_y,num_iterations=20000, print_cost = True, layer_dims=layer_dims)

    #plot the decision boundary
    #plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        #plt.title("Decision boundary for hidden layer size " + str(4))

    # Accuracy
    #predictions = predict(parameters, X)
    #print(f'Accuracy: {float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)* 100)} % ')

    

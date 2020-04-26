import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import tensorflow as tf

from utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):

    xavier = tf.initializers.GlorotUniform()
    
    W1 = tf.Variable(xavier(shape=(n_h, n_x)))
    W1 = tf.cast(W1, tf.float64)

    b1 = tf.zeros([n_h, 1],dtype=tf.float64)

    W2 = tf.Variable(xavier(shape=(n_y, n_h)))
    W2 = tf.cast(W2, tf.float64)
    b2 = tf.zeros([n_y, 1],dtype=tf.float64)

    
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    

    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.tanh(Z1)

    Z2 = tf.matmul(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    
    return A2, cache
    


def compute_cost(A2, Y, parameters):
    m = Y.shape[1]

    cost = (-1/m) * (tf.matmul(Y, tf.transpose(tf.math.log(A2))) + tf.matmul(1-Y, tf.transpose(tf.math.log(1-A2))))
    cost = tf.cast(cost, tf.float64)
                  
    return cost


def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    A1 = cache['A1']
    A2 = cache['A2']
    
    dZ2 = A2 - Y
    dW2 = (1 / m) * tf.matmul(dZ2, tf.transpose(A1))
    db2 = (1 / m) * tf.reduce_sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = tf.matmul(tf.transpose(W2), dZ2) * (1 - tf.math.pow(A1, 2))
    dW1 = (1 / m) * tf.matmul(dZ1, tf.transpose(X))
    db1 = (1 / m) * tf.reduce_sum(dZ1, axis=1, keepdims=True)
    
    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    
    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
        
    }
    
    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost = False):

    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    X = tf.cast(X, tf.float64)
    Y = tf.cast(Y, tf.float64)
    
    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
      
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward Propagation
        A2, cache = forward_propagation(X, parameters)
        
        # cost function
        cost = compute_cost(A2, Y, parameters)
        
        # Backpropagation
        grads = backward_propagation(parameters, cache, X, Y)
        
        # parameter update
        parameters = update_parameters(parameters, grads)
        
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

    X, Y = load_planar_dataset()

    parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost = True)

    #plot the decision boundary
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision boundary for hidden layer size " + str(4))

    # Accuracy
    predictions = predict(parameters, X)
    print(f'Accuracy: {float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)* 100)} % ')

    
    # Tunung different size of hidden layer
    plt.figure(figsize=(16, 32))
    hidden_layer_size = [1, 2, 3, 4, 5, 20, 50]
    for i, n_h, in enumerate(hidden_layer_size):
        plt.subplot(5, 2, i+1)
        plt.title('Hidden layer of size %d' %n_h)
        parameters = nn_model(X, Y, n_h, num_iterations = 5000)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        predictions = predict(parameters, X)
        print(f'Accuracy: {float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)* 100)} % ')
        
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from main import Main

class Model:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.main = Main()

    def fit(self, X, Y, learning_rate = 0.0007, num_iterations = 2000, print_cost = True):
        costs = []      
        L = len(self.layer_dims)   

        # Parameters initialization.
        parameters = self.main.initialize_parameters(self.layer_dims)
        
        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: 
            AL, caches = self.main.forward_propagation(X, parameters)
            
            # Compute cost.
            cost = self.main.compute_cost(AL, Y)
        
            # Backward propagation.
            grads =  self.main.backward_propagation(AL, Y, caches, parameters)
        
    
            # Update parameters.
            parameters = self.main.update_parameters(parameters, grads, learning_rate)
                    
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)
            
        
         # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        return parameters


    def predict(self, X, y, parameters):
        
        m = X.shape[1]
        n = len(parameters) // 2 
        p = np.zeros((1,m))
       
        # Forward propagation
        probas, caches = self.main.forward_propagation(tf.cast(X, tf.float32), parameters)

        
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0

       # print("Accuracy: " + str(np.sum((p == y) / n)))
            
        return p






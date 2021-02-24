import matplotlib.pyplot as plt
import numpy as np

from nn import NN

class Main:
    def __init__(self, layers_dims):
        self.nn = NN()
        self.layers_dims = layers_dims


    def L_layer_model(self, X, Y, layers_dims, learning_rate = 0.0007,num_epochs = 2000, print_cost = True,lambd=0):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
        
        Arguments:
        X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """ 
        costs = []      
        L = len(layers_dims)                     
        # Parameters initialization.
        parameters = self.nn.initialize_parameters(layers_dims)

        # Loop (gradient descent)
        for i in range(num_epochs):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = self.nn.L_model_forward(X, parameters)
            # Compute cost.
            if lambd == 0:
                cost = self.nn.compute_cost(AL, Y)
            else:
                cost = self.nn.compute_cost_with_regularization(AL, Y, parameters, lambd)

            # Backward propagation.
            grads =  self.nn.L_model_backward(AL, Y,caches ,lambd, parameters)
            # Update parameters
            parameters = self.nn.update_parameters(parameters, grads, learning_rate)

            # Print the cost every 1000 epoch
            if print_cost and i % 100 == 0:
                print ("Cost after epoch %i: %f" %(i, cost))
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
        """
        This function is used to predict the results of a  L-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        
        m = X.shape[1]
        n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        
        # Forward propagation
        probas, caches = self.nn.L_model_forward(X, parameters)

        
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        print("Accuracy: "  + str(np.sum((p == y)/m)))
            
        return p



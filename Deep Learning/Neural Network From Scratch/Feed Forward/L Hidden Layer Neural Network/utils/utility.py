import numpy as np


class Utils:

    def sigmoid(self, z):
        a = 1 / (1 + np.exp(-z))
        cache = z

        return a, cache

    def relu(self, z):
        a = np.maximum(0, z)
        cache = z

        return a, cache

    def relu_backward(self, da, cache):
        z = cache
        dz = np.array(da, copy=True)  # just converting dz to a correct object.
        # When z <= 0, you should set dz to 0 as well.
        dz[z <= 0] = 0

        return dz

    def sigmoid_backward(self, da, cache):
        z = cache
        s = 1 / (1 + np.exp(-z))
        dz = da * s * (1 - s)

        return dz	

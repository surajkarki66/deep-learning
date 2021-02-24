import numpy as np
weights = np.array([0.1, 0.2, 0])


def neural_network(input, weights):
    pred = np.dot(input, weights)
    return pred


toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])
input = np.array([toes[0], wlrec[0], nfans[0]])
pred = neural_network(input, weights)
print(pred)

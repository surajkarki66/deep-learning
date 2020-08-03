
# Single input( One data point)
weight = 0.1


def neural_network(weight, input):
    prediction = input * weight
    return prediction


number_of_toes = [8.5,  9.5, 10, 11]  # inputs
input = number_of_toes[0]
pred = neural_network(weight, input)
print(pred)

# Multiple Input( Multiple datapoint )( 3 data points)
# No of weights is equal to the number of data points in a single neuron

weights = [0.1, 0.2, 0]


def neural_network(input, weights):
    pred = w_sum(input, weights)
    return pred


def w_sum(input, weights):
    output = 0
    for i in range(len(input)):
        output += (input[i] * weights[i])
    return output


toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

input = [toes[0], wlrec[0], nfans[0]]

pred = neural_network(input, weights)
print(pred)

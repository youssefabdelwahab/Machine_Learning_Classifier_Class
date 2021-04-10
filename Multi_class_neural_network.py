import math
import numpy as np

class artificial_neural_network():

    def __init__(self , num_inputs = 3 , hidden_layers=[4,4] , outputs= 1):
        self.num_input = num_inputs
        self.hidden_layers = hidden_layers
        self.outputs = outputs

        layers = [num_inputs] + hidden_layers + [outputs]


        weights = []
        for i in range(len(layers - 1)):
            w = np.random.rand(layers[i] , layers[i + 1])
            weights.append(w)
        self.weights = weights

        biases = []
        for i in range(len(layers - 1)):
            b = np.random.rand(layers[i] , layers[i + 1])
            biases.append(b)
        self.biases = biases

        derivatives = []
        for i in range(len(layers - 1)):
            d = np.zeros(layers[i] , layers[i + 1])
            derivatives.append(d)
        self.derivatives = derivatives

        activation_data = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activation_data.append(a)
        self.activation_data = activation_data


    def forward_propogation(self, inputs):

        features = inputs

        self.activation_data[0] = features

        for i, (w, b) in zip(self.weights - 1, self.biases - 1):
            net_inputs = np.dot(features , w)  + b

            features = self.sigmoid(net_inputs)

            self.activation_data[i + 1] = features

            for (g,x) in zip(self.weights[-1] , self.biases[-1]):
                net_inputs = np.dot(features , g) + x

                features = net_inputs

                self.activation_data = features
        return features


    def back_propogation(self, error):














    def sigmoid(self , X):
        return 1 / (1 + np.exp(-X))

    def sigmoid_prime(self, X):
        return X * (1.0 - X)

    def _mean_squared_error(self, target, output):
        return np.average((target - output) ** 2)

    def mse_prime(self, target ,  output):
        return target - output



from numpy import dot 
from numpy.random import uniform

class Perceptron():
    def __init__(self, inputs, outputs, learning_rate, activation_function):
        self.learning_rate = learning_rate
        self.inputs = inputs
        self.outputs = outputs
        self.weights = uniform(size=2)
        self.bias = uniform()
        self.actv = activation_function

    def z_sum(self, inputs):
        return dot(inputs, self.weights) + self.bias

    def recalculate_weights(self, inpt, epsilon):
        return self.weights - inpt*epsilon*self.learning_rate

    def recalculate_bias(self, epsilon):
        return self.bias - epsilon*self.learning_rate

    def predict(self, inputs):
        z = self.z_sum(inputs)
        return self.actv(z) 

    def train(self, epochs):
        epsilon_record = []

        for epoch in range(epochs):
            cumulative_epsilon = 0

            for x, y in zip(self.inputs, self.outputs):
                y_pred = self.predict(x)
                epsilon = y_pred - y[0]     # Take the single element of the array 

                cumulative_epsilon += abs(epsilon)

                self.weights = self.recalculate_weights(x, epsilon)
                self.bias = self.recalculate_bias(epsilon)
            
            epsilon_record.append(cumulative_epsilon / len(self.inputs[:, 0]))

        return (self.weights, epsilon_record)
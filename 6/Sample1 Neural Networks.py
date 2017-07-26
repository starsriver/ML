# -*- coding: utf-8 -*-
from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, trainSetInputs, trainSetOutputs, numberOftrainIterations):
        for iteration in range(numberOftrainIterations):
            outputs = self.think(trainSetInputs)
            error = trainSetOutputs - outputs
            adjustment = dot(
                trainSetInputs.T, error * self.__sigmoid_derivative(outputs))
            self.synaptic_weights += adjustment

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":
    neuralNetwork = NeuralNetwork()

    print("随机的初始突触权重：")
    print(str(neuralNetwork.synaptic_weights))

    trainSetInputs = array([
        [0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]
        ])

    trainSetOutputs = array([
        [0, 1, 1, 0]
        ]).T
    neuralNetwork.train(trainSetInputs, trainSetOutputs, 1000000)

    print("训练后的突触权重：")
    print(str(neuralNetwork.synaptic_weights))

    print("新的测试数据：[1, 0, 0] -> ?")
    print(neuralNetwork.think(array([1, 0, 0])))

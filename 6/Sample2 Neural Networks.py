# -*- coding: utf-8 -*-
from numpy import exp, array, random, dot


class NeuronLayer():
    def __init__(self, numberOfNeurons, numberOfInputsPerNeuron):
        self.synapticWeights = 2 * random.random(
            (numberOfInputsPerNeuron, numberOfNeurons)) - 1


class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, trainSetInputs, trainSetOutputs, trainTimes):
        for time in range(trainTimes):
            outputFromLayer1, outputFromLayer2 = self.think(trainSetInputs)

            layer2Error = trainSetOutputs - outputFromLayer2
            layer2Delta = layer2Error * self.__sigmoid_derivative(
                outputFromLayer2)

            layer1Error = layer2Delta.dot(self.layer2.synapticWeights.T)
            layer1Delta = layer1Error * self.__sigmoid_derivative(
                outputFromLayer1)
            layer1Adjustment = trainSetInputs.T.dot(layer1Delta)
            layer2Adjustment = outputFromLayer1.T.dot(layer2Delta)

            self.layer1.synapticWeights += layer1Adjustment
            self.layer2.synapticWeights += layer2Adjustment

    def think(self, inputs):
        outputFromLayer1 = self.__sigmoid(
            dot(inputs, self.layer1.synapticWeights))
        outputFromLayer2 = self.__sigmoid(
            dot(outputFromLayer1, self.layer2.synapticWeights))
        return outputFromLayer1, outputFromLayer2

    def printWeight(self):
        print("Layer1 (4 neurons, each with 3 input):")
        print(str(self.layer1.synapticWeights))
        print("Layer2 (1 neurons, each with 4 input):")
        print(str(self.layer2.synapticWeights))


if __name__ == "__main__":
    random.seed(1)

    layer1 = NeuronLayer(4, 3)
    layer2 = NeuronLayer(1, 4)
    neuralNetwork = NeuralNetwork(layer1, layer2)

    print("随机的初始突触权重：")
    neuralNetwork.printWeight()

    trainSetInputs = array([
        [0, 0, 1], [0, 1, 1], [1, 0, 1],
        [0, 1, 0], [1, 0, 0], [1, 1, 1],
        [0, 0, 0]
        ])

    trainSetOutputs = array([
        [0, 1, 1, 1, 1, 0, 0]
        ]).T
    neuralNetwork.train(trainSetInputs, trainSetOutputs, 1000000)

    print("训练后的突触权重：")
    neuralNetwork.printWeight()

    print("新的测试数据：[1, 1, 0] -> ?")
    hiddenState, output = neuralNetwork.think(array([1, 1, 0]))
    print(str(output))

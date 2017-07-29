# -*- coding: utf-8 -*-
from NeuralNetwork import NeuralNetwork
import numpy as np

nn = NeuralNetwork([2, 2, 1], "tanh")
X = np.array([[0, 0], [0, 1], [1, 0], [0, 0]])
Y = np.array([0, 1, 1, 0])
nn.fit(X, Y)

for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    print(i, nn.predict(i))

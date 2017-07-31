# -*- coding: utf-8 -*-
import numpy as np
import random


def gradientDescent(x, y, theta, alpha, m, numIterations):
    x_trans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration: %d | Cost: %f" % (i, cost))
        gradient = np.dot(x_trans, loss) / m
        theta = theta - alpha * gradient
    return theta


def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    for i in range(0, numPoints):
        x[i][0] = 1
        x[i][1] = i
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y


x, y = genData(100, 25, 10)
m, n = np.shape(x)
numIterations = 100000
alpha = 0.0005
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)

print(theta)

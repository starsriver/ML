import numpy as np


def getPearsonCorrelationCoefficient(x, y):
    length = np.min((len(x), len(y)))
    x = x[:length]
    y = y[:length]
    _x = np.mean(x)
    _y = np.mean(y)
    up = 0
    downX = 0
    downY = 0
    for i in range(length):
        tempX = x[i] - _x
        tempY = y[i] - _y
        up += tempX * tempY
        downX += tempX ** 2
        downY += tempY ** 2
    return up / np.sqrt(downX * downY)


def polyfit(x, y, degree):
    results = {}
    coffes = np.polyfit(x, y, degree)
    results["polynomial"] = coffes.tolist()
    p = np.poly1d(coffes)
    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results["determination"] = ssreg / sstot
    return results

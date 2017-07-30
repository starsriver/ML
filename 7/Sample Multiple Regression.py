# -*- coding: utf-8 -*-
from numpy import genfromtxt
from sklearn import linear_model

dataPath = r"./Delivery.csv"
deliveryData = genfromtxt(dataPath, delimiter=',')
print "data"
print deliveryData

X = deliveryData[:, :-1]
Y = deliveryData[:, -1]

print("X: ", X)
print("Y: ", Y)

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print("coefficients: ", regr.coef_)
print("intercept: ", regr.intercept_)

xPred = [102, 6]
yPred = regr.predict(xPred)

print("oredicted y: ", yPred)

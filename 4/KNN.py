# -*- coding: utf-8 -*-
import csv
import random
import math
import operator
import os

def loadDataSet(fileName, split, trainingSet=[], testSet=[]):
    with open(fileName, "rb") as csvFile:
        lines = csv.reader(csvFile)
        dataSet = list(lines)
        for x in range(len(dataSet) - 1):
            for y in range(4):
                dataSet[x][y] = float(dataSet[x][y])
            if random.random() < split:
                trainingSet.append(dataSet[x])
            else:
                testSet.append(dataSet[x])


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow(instance1[x] - instance2[x], 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        distance = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], distance))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(
        classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


def main():
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataSet(r"../DataSet/iris.data.txt", split, trainingSet, testSet)

    print("Train Set: " + repr(len(trainingSet)))
    print("Test Set: " + repr(len(testSet)))

    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print(
            "> predicted= " + repr(result) + ", actual= " +
            repr(testSet[x][-1])
            )
    accuracy = getAccuracy(testSet, predictions)

    print("Accuracy: " + repr(accuracy) + " %")


if __name__ == '__main__':
    print(os.getcwd())
    main()

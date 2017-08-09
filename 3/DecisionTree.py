# -*- coding: utf-8 -*-
from math import log


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfaceing', 'flippers']
    return dataSet, labels


# 计算香农熵
def calcShannonEnt(dataSet):
    length = len(dataSet)
    labelCounts = {}
    # 统计频率
    for data in dataSet:
        currentLabel = data[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 计算熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / length
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


# 按照特征属性值对数据集进行划分
def splitDataSet(dataSet, axis, value):
    newDataSet = []
    for data in dataSet:
        if data[axis] == value:
            reducedFeatVec = data[:axis]
            reducedFeatVec.extend(data[axis + 1:])
            newDataSet.append(reducedFeatVec)
    return newDataSet


# 选择最好的元素进行划分
def chooseBestFeatureToSplit(dataSet):
    # 类别标签不参与划分
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # 对每个特征属性都尝试划分，找到最好的划分元素
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 进行多数表决
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    return max(classCount)


# 递归构建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 类别相同时则划分停止
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 所有特征属性已经用完，采用多数表决确定节点类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def test():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfaceing', 'flippers']
    myTree = createTree(dataSet, labels)
    print myTree


if __name__ == '__main__':
    test()

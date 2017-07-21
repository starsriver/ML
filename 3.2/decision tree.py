# -*- coding: utf-8 -*-

from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree

allElectronicsData = open(r"./ALLElectronics.csv", "rb")
reader = csv.reader(allElectronicsData)
headers = reader.next()

# print(headers)

featrueList = []
labelList = []

for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]
    featrueList.append(rowDict)

# print(featrueList)

vec = DictVectorizer()
dummyX = vec.fit_transform(featrueList).toarray()

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)

clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(dummyX, dummyY)

with open(r"./ALLElectronics.dot", "w") as f:
    f = tree.export_graphviz(
        clf, feature_names=vec.get_feature_names(), out_file=f)

oneRowX = dummyX[0, :]

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0

predictedY = clf.predict(newRowX)
print(str(clf.predict([0, 0, 1, 0, 1, 1, 0, 0, 1, 0])))
print(str(predictedY))

# dot -Tpdf dotfile -o out.pdf
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 12:46:51 2018

@author: Ayushi Jain
"""
import numpy as np
from sklearn import svm
from sklearn import neural_network
from sklearn import neighbors
from sklearn import ensemble
from sklearn.metrics import accuracy_score

clf_svm = svm.SVC()
clf_nn = neural_network.MLPClassifier()
clf_knn = neighbors.KNeighborsClassifier()
clf_rf = ensemble.RandomForestClassifier()
clf_ab = ensemble.AdaBoostClassifier()
clf_b = ensemble.BaggingClassifier()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf_svm = clf_svm.fit(X, Y)
clf_nn = clf_nn.fit(X, Y)
clf_knn = clf_knn.fit(X, Y)
clf_rf = clf_rf.fit(X,Y)
clf_ab = clf_ab.fit(X,Y)
clf_b = clf_b.fit(X,Y)

prediction_svm = clf_svm.predict(X)
prediction_nn = clf_nn.predict(X)
prediction_knn = clf_knn.predict(X)
prediction_rf = clf_rf.predict(X)
prediction_ab = clf_ab.predict(X)
prediction_b = clf_b.predict(X)
# compare their results and print the best one!

acc_SVM = accuracy_score(Y, prediction_svm)*100
print("Accuracy for SVM: {}".format(acc_SVM))

acc_KNN = accuracy_score(Y, prediction_knn)*100
print("Accuracy for KNN: {}".format(acc_KNN))

acc_NN = accuracy_score(Y, prediction_nn)*100
print("Accuracy for NN: {}".format(acc_NN))

acc_rf = accuracy_score(Y, prediction_rf)*100
print("Accuracy for Random Forests: {}".format(acc_rf))

acc_ab = accuracy_score(Y, prediction_ab)*100
print("Accuracy for AdaBoost Classifier: {}".format(acc_ab))

acc_b = accuracy_score(Y, prediction_b)*100
print("Accuracy for Bagging Classifier: {}".format(acc_b))

#choosing the most accurate classifier
clf = [acc_SVM, acc_KNN, acc_NN, acc_rf,acc_ab,acc_b]
index = [i for i, x in enumerate(clf) if x == max(clf)]
cls_dict = {0: 'SVM', 1: 'K Nearest Neighbors', 2: 'Neural Networks', 3: 'Random Forests', 4:'AdaBoost Classifier', 5:'Bagging Classifier'}
print("The best gender classifiers are:", end=" ")
for i in index:
    print(cls_dict[i], end=", ")


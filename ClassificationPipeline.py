__author__ = 'DougGreiwe'

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifierCV, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, classification_report
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing, svm, datasets, cross_validation
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from time import time

import numpy as np
import pandas as pd

# imports iris dataset
iris = datasets.load_iris()
x_cols = iris.data[:, :2]
y_col = iris.target
X_iris = x_cols
y_iris = y_col

#X = np.array
#y_one = np.array([y_col])
#y = y_one.ravel()

tuned_parameters = [
    {'kernel' : ['linear'], 'C' : [1, 10, 100, 1000]},
    {'criterion' : ['gini', 'entropy'], 'splitter' : ['best', 'random'], 'max_features' : [None]},
    {'n_estimators' : [10, 50, 100, 500], 'criterion' : ['gini', 'entropy'], 'max_features' : [None]},
    {'n_estimators' : [50,100,500], 'learning_rate' : [.5,.75,1,1.25,2]},
    {'n_neighbors' : [5, 10, 25], 'weights' : ['uniform', 'distance'],
     'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size' : [10,30,50], 'p' : [1,2,3]},
    {'shrink_threshold' : [None, .01, .1,.5]}
    ]

learning_funcs = [
    svm.SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    KNeighborsClassifier(),
    NearestCentroid()
    ]

for i in range(0, len(learning_funcs)):

    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.5, random_state=0)

    clf = GridSearchCV(learning_funcs[i], tuned_parameters[i], cv = 10)
    # sklearn_pandas does not support estimators accepting y vector with labels
    t0 = time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print
    print ("Learning Function:")
    print (learning_funcs[i])
    print
    print ("Best Estimator:")
    print (clf.best_estimator_)
    print
    print ("Parameters:")
    print (clf.param_grid)
    print
    print ("Best Parameters:")
    print (clf.best_params_)
    print
    print ("Best Score:")
    print (clf.best_score_)
    print
    print ("Classification Report:")
    print (classification_report(y_test, y_pred))
    print
    print ("Done in %0.3fs" % (time() - t0))
    #The precision is the ability of the classifier not to label as positive a sample that is negative
    #The recall is intuitively the ability of the classifier to find all the positive samples
    # The support is the number of occurrences of each class in y_true or the testing data
    plt.figure()
    plt.clf()
    plt.scatter(X_train[:,0], X_train[:,1], c= 'Y', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title(learning_funcs[i])
    plt.show()
    print ("_________________________________________________________________________________")



__author__ = 'DougGreiwe'

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import  LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
from sklearn import preprocessing, cross_validation
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from time import time


Location = r'/Users/DougGreiwe/Desktop/MachineLearning/fredgraph.xlsx'
df = pd.read_excel(Location)

x_cols = ['GDP']
y_col = ['Percent_Debt']
X = np.array(df[x_cols])
y_one = np.array(df[y_col])
y = y_one.ravel()

regr_tuned_parameters = [
    {},
    {},
    {'fit_intercept' : [True, False]},
    {},
    {'n_estimators' : [10, 50, 100, 500], 'max_features' : [None, 'auto']}
    ]

regr_learning_funcs = [
    SGDRegressor(),
    Lasso(),
    LinearRegression(),
    Ridge(),
    RandomForestRegressor()
    ]

for i in range(0, len(regr_learning_funcs)):
    t0 = time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)
    clf = GridSearchCV(regr_learning_funcs[i], regr_tuned_parameters[i], cv=10)
    clf.fit(X_train, y_train)
    y_pred, y_pred2 = clf.predict(X_test), clf.predict(X_train)
    print
    print ("Learning Function:")
    print (regr_learning_funcs[i])
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
    print ("Mean Squared Error:")
    print (mean_squared_error(y_test, y_pred))
    print
    print ("Mean Absolute Error:")
    print (mean_absolute_error(y_test, y_pred))
    print
    print ("R Squared Score:")
    print (r2_score(y_test, y_pred))
    print
    print ("Done in %0.3fs" % (time() - t0))
    plt.figure()
    plt.scatter(X_test, y_test, color = 'red', label = "Testing Samples")
    plt.scatter(X_train, y_train, color ='blue', label = "Training Samples")
    plt.plot(X_test, y_pred, color = 'red', label = "Total Public Debt from Testing Data", linewidth = 1, linestyle = '--')
    plt.plot(X_train, y_pred2, color = 'blue', label = "Total Public Debt from Training Data", linewidth = 1, linestyle = '--')
    plt.xlabel("GDP")
    plt.ylabel("Total Public Debt as Percent of GDP")
    plt.title(regr_learning_funcs[i])
    plt.legend()
    plt.show()
    print ("_________________________________________________________________________________")

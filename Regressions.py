__author__ = 'DougGreiwe'

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import  LinearRegression, Ridge, RidgeCV, Lasso, SGDRegressor, ElasticNet, lasso_path, enet_path
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, explained_variance_score
from sklearn import preprocessing, cross_validation, datasets
from itertools import cycle
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import DataFrame, Series
from time import time
Location = r'http://chart.finance.yahoo.com/table.csv?s=GOOG&a=11&b=2&c=2010&d=11&e=2&f=2016&g=w&ignore=.csv'
Location2 = r'http://chart.finance.yahoo.com/table.csv?s=YHOO&a=11&b=2&c=2010&d=11&e=2&f=2016&g=w&ignore=.csv'
df = pd.read_csv(Location)
df2 = pd.read_csv(Location2)
x_cols = ['Close']
y_col = ['Close']
X = np.array(df[x_cols])
y_one = np.array(df2[y_col])
y = y_one.ravel()
#diabetes = datasets.load_diabetes()
#X = diabetes.data
#y = diabetes.target
#Split the data into training/testing parts
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5, random_state=0)
t0 = time()
print("")
print("Tuning Hyper-Parameters...")
print("")
lr_tuned_parameters = [{'fit_intercept' : [True, False]}]
#The class ElasticNetCV can be used to set the parameters alpha (\alpha) and l1_ratio (\rho) by cross-validation.
print("Fitting Trained Data...")
print("")
clf1 = GridSearchCV(LinearRegression(), lr_tuned_parameters, cv=5)
clf1.fit(X_train, y_train)
y_true1, y_pred1 = clf1.predict(X_test), clf1.predict(X_train)
clf2 = Ridge(fit_intercept = False)
clf2.fit(X_train, y_train)
y_true2, y_pred2 = clf2.predict(X_test), clf2.predict(X_train)
clf3 = Lasso(alpha=0.1)
clf3.fit(X_train, y_train)
y_true3, y_pred3 = clf3.predict(X_test), clf3.predict(X_train)
clf4 = ElasticNet(alpha=0.1, l1_ratio=0.7)
clf4.fit(X_train, y_train)
y_true4, y_pred4 = clf4.predict(X_test), clf4.predict(X_train)
print("Linear Regression Scores:")
print("Mean Squared Error: %.4f" % (mean_squared_error(y_true1, y_pred1)))
print("Mean Absolute Error: %.4f" % (mean_absolute_error(y_true1, y_pred1)))
print("Median Absolute Error: %.4f" % (median_absolute_error(y_true1, y_pred1)))
print("R Squared: %.4f" % (r2_score(y_true1, y_pred1)))
print("Variance: %.4f" % (explained_variance_score(y_true1, y_pred1)))
print("")
print("Ridge Scores:")
#print("Ridge Coefficient: %.4f" % (clf2.coef_))
#print("Ridge Alpha Value: %.4f" % (clf2.alpha_))
print("Mean Squared Error: %.4f" % (mean_squared_error(y_true2, y_pred2)))
print("Mean Absolute Error: %.4f" % (mean_absolute_error(y_true2, y_pred2)))
print("Median Absolute Error: %.4f" % (median_absolute_error(y_true2, y_pred2)))
print("R Squared: %.4f" % (r2_score(y_true2, y_pred2)))
print("Variance: %.4f" % (explained_variance_score(y_true2, y_pred2)))
print("")
print("Lasso Scores:")
print("Mean Squared Error: %.4f" % (mean_squared_error(y_true3, y_pred3)))
print("Mean Absolute Error: %.4f" % (mean_absolute_error(y_true3, y_pred3)))
print("Median Absolute Error: %.4f" % (median_absolute_error(y_true3, y_pred3)))
print("R Squared: %.4f" % (r2_score(y_true3, y_pred3)))
print("Variance: %.4f" % (explained_variance_score(y_true3, y_pred3)))
print("")
print("Lasso Scores:")
print("Mean Squared Error: %.4f" % (mean_squared_error(y_true4, y_pred4)))
print("Mean Absolute Error: %.4f" % (mean_absolute_error(y_true4, y_pred4)))
print("Median Absolute Error: %.4f" % (median_absolute_error(y_true4, y_pred4)))
print("R Squared: %.4f" % (r2_score(y_true4, y_pred4)))
print("Variance: %.4f" % (explained_variance_score(y_true4, y_pred4)))
print("")

print("Done in %0.3fs" % (time() - t0))
print("")
#Linear Regression
#plt.figure()
#plt.scatter(X_test, y_test, color='red', label="Testing Samples")
#plt.scatter(X_train, y_pred1, color='blue', label="Training Samples")
#plt.plot(X_train, y_pred1, color='black', linewidth=.5, linestyle='--')
#plt.xlabel("Opening Stock Price")
#plt.ylabel("Closing Stock Price")
#plt.title("Linear Regression of Stock Prices Feb. 2010 - Today (Monthly)")
#plt.legend()
#plt.show()
n_alphas = 200
alphas = np.logspace(-3, -0.5, n_alphas)
coefs = []
for a in alphas:
    clf2.set_params(alpha=a)
    clf2.fit(X_train, y_train)
    coefs.append(clf2.coef_)
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('Alpha')
plt.ylabel('Weights')
plt.title('Ridge Coef. as Function of Regularization \n')
plt.axis('tight')
plt.show()
#Lasso vs. Elastic Net:
#This plot allows us to see how the coefficients react when being forced to become positive (in both lasso
#and elastic net). You can see the effects on the ridge, lasso, and elastic net when using multiple features.
X /= X.std(axis=0)
eps = 5e-3
alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_pred3, eps, fit_intercept=False)
alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(X_train, y_pred3, eps, positive=True, fit_intercept=False)
alphas_enet, coefs_enet, _ = enet_path(X_train, y_pred4, eps=eps, l1_ratio=0.8, fit_intercept=False)
alphas_positive_enet, coefs_positive_enet, _ = enet_path(X_train, y_pred4, eps=eps, l1_ratio=0.8, positive=True, fit_intercept=False)
plt.figure(1)
ax = plt.gca()
colors = cycle(['b', 'r', 'g', 'c', 'k'])
neg_log_alphas_lasso = -np.log10(alphas_lasso)
neg_log_alphas_enet = -np.log10(alphas_enet)
for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2 = plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)
plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and Elastic-Net Paths')
plt.legend((l1[-1], l2[-1]), ('Lasso', 'Elastic-Net'), loc='lower left')
plt.axis('tight')
plt.figure(2)
ax = plt.gca()
neg_log_alphas_positive_lasso = -np.log10(alphas_positive_lasso)
for coef_l, coef_pl, c in zip(coefs_lasso, coefs_positive_lasso, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2 = plt.plot(neg_log_alphas_positive_lasso, coef_pl, linestyle='--', c=c)
plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and positive Lasso')
plt.legend((l1[-1], l2[-1]), ('Lasso', 'positive Lasso'), loc='lower left')
plt.axis('tight')
plt.figure(3)
ax = plt.gca()
neg_log_alphas_positive_enet = -np.log10(alphas_positive_enet)
for (coef_e, coef_pe, c) in zip(coefs_enet, coefs_positive_enet, colors):
    l1 = plt.plot(neg_log_alphas_enet, coef_e, c=c)
    l2 = plt.plot(neg_log_alphas_positive_enet, coef_pe, linestyle='--', c=c)
plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Elastic-Net and positive Elastic-Net')
plt.legend((l1[-1], l2[-1]), ('Elastic-Net', 'positive Elastic-Net'),
           loc='lower left')
plt.axis('tight')
plt.show()
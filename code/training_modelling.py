# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 15:29:39 2022

@author: hthiam
"""

#importaing librairies
import numpy as np


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score


############################################################################## LINEAR REGRESSION MODEL

def linear_regression(X_train, X_test, Y_train, Y_test):
    # training
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    # evaluate in the trainint set
    y_train_predict = model.predict(X_train)
    rmse_train = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
    r2_train = r2_score(Y_train, y_train_predict)
    
    print('Model performance based on the training set')
    print('--------------------------------------')
    print('The root mean square error ist {}'.format(rmse_train))
    print('The R2 score is {}'.format(r2_train))
    print('\n')
    
    # model evaluation for testing set
    y_test_predict = model.predict(X_test)
    rmse_test = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
    r2_test = r2_score(Y_test, y_test_predict)
    
    print('Model performance based on the training set')
    print('--------------------------------------')
    print('The root mean square error ist {}'.format(rmse_test))
    print('The R2 score is {}'.format(r2_test))
    print('\n')
    
    result = {"rmse_test_lr": rmse_test, "r2_test_lr": r2_test}
    return (result, y_test_predict)

############################################################################## RANDOM FOREST

def random_forest(X_train, X_test, Y_train, Y_test):
    #initializing the model
    model = RandomForestRegressor(random_state = 0)
    # fitting the model
    model.fit(X_train, Y_train)
    
    #predicting the target of the test set
    y_test_predict = model.predict(X_test)
    
    # model evaluation for testing set
    rmse_test = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
    r2_test = r2_score(Y_test, y_test_predict)

    print('Model performance based on the training set')
    print('--------------------------------------')
    print('The root mean square error ist {}'.format(rmse_test))
    print('The R2 score is {}'.format(r2_test))
    print('\n')
    
    result = {"rmse_test_rf": rmse_test, "r2_test_fr": r2_test}
    return (result, y_test_predict)

















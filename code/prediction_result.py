# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 18:58:16 2022

@author: hthiam
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from training_modelling import*



############################################################################## Importing data cleaning
work = r'C:/Users/hthiam/Desktop/python ecole/'
path = 'data_cleaning.csv'
books = pd.read_csv(path, sep = ";")

############################################################################## Spliting data 
# Get features
features = ['authors','language_code','num_pages', 'ratings_count','text_reviews_count','publisher', 'YEAR']
X = books[features]
# Get target
target = 'average_rating'
Y = books[target]

#Spliting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# Get result
random_prediction = random_forest(X_train, X_test, Y_train, Y_test)
linear_regression_prediction = linear_regression(X_train, X_test, Y_train, Y_test)

# Create table to vizualise in power BI
prediction = np.transpose(np.array([list(Y_test), random_prediction[1],linear_regression_prediction[1]]))
result = pd.DataFrame(prediction, columns = ["Y_test", "Y_test_rf", "Y_test_lr"])
path_export = 'prediction.csv'
result.to_csv(work +path_export,
            sep=';',
            na_rep='',
            header=True,
            index=True,
            decimal=',',
            encoding='utf-8-sig',
            date_format='%d/%m/%Y')

#Get the rmse
kpi1 = random_prediction[0]
kpi2 = linear_regression_prediction[0]
kpi1["rmse_test_lr"] = kpi2["rmse_test_lr"]
kpi1['r2_test_lr'] = kpi2['r2_test_lr']
kpi = list(kpi1.keys())
value = list(kpi1.values())
kpi = pd.DataFrame(np.transpose(np.array([kpi, value])), columns =["kpi", "value"])
path_export = 'kpi.csv'
kpi.to_csv(path_export,
            sep=';',
            na_rep='',
            header=True,
            index=True,
            decimal=',',
            encoding='utf-8-sig',
            date_format='%d/%m/%Y')
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:56:26 2022

@author: hthiam
"""

#importaing librairies
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing 
import seaborn as sns
import seaborn as sns; sns.set_theme()
import statsmodels.api 

############################################################################## Importing data cleaning
work = r'C:/Users/hthiam/Desktop/python ecole/'
path = 'data_cleaning.csv'
books = pd.read_csv(work +path, sep = ";")

############################################################################## CORRELATION
# correlation matrix
mat_corr = books.corr() 
heatmap = sns.heatmap(mat_corr, square=True)
mat_corr["average_rating"]
# export matrix corelation 
path_export = 'mat_corr.csv'
mat_corr.to_csv(work + path_export,
            sep=';',
            na_rep='',
            header=True,
            index=False,
            decimal=',',
            encoding='utf-8-sig',
            date_format='%d/%m/%Y')

############################################################################## Anova à plusieurs facteur
#books.rename(columns={'  num_pages': 'num_pages'}, inplace=True)
# instancie le modèle
label = "average_rating"
features =  list(books.columns)
del features[0:3:2]
vec = label + "~authors"
for col in features[1:]:
    vec += "+" + col
result = statsmodels.formula.api.ols(vec, data = books).fit() 
# table anova
anova = statsmodels.api.stats.anova_lm(result)
path_anova = 'anova.csv'
anova.to_csv(work + path_anova,
            sep=';',
            na_rep='',
            header=True,
            index=True,
            decimal=',',
            encoding='utf-8-sig',
            date_format='%d/%m/%Y')
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:57:32 2022

@author: hthiam
"""

#importaing librairies
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing 

############################################################################## Importing data
work = r'C:/Users/hthiam/Desktop/python ecole/'
path = 'books.csv'
books = pd.read_csv(work+path, sep =",", on_bad_lines='skip')

##### Cleaning the date column

# Creating year colonne
date = list(books["publication_date"])
# Get the date
year = [int(y[-4:]) for y in date]
year_encoding = []
for y in year :
    if 1900<=y and y <= 1920:
        year_encoding.append(0)
    elif 1920<y and y <= 1940:
        year_encoding.append(1)
    elif 1940<y and y <= 1960:
        year_encoding.append(2)
    elif 1960<y and y <= 1980:
        year_encoding.append(3)
    elif 1980<y and y <= 2000:
        year_encoding.append(4)
    else:
        year_encoding.append(5)
# Add the date in the data
books["YEAR_ENCODING"] = year_encoding
books["YEAR"] = year

##### delete the bads lines in the avarage_rating and the title, isbn, publication date columns
bad_rating =[12224, 16914, 22128, 34889]
for bl in bad_rating:
    books = books[books["average_rating"]!=bl]
books = books.drop(["title","isbn", "publication_date"], axis = 1) 
##### ENCODING 
# copy thr books data
df = books.copy()

def encoding(data, columns):
    """
    data : the data frame
    columns : list of column that we encoding
    """
    label_encoder = preprocessing.LabelEncoder()
    for col in columns:
        df[col]= label_encoder.fit_transform(df[col])
    return data
data = df
columns =['authors', 'language_code' , 'publisher',  ]   
data_cleaning = encoding(data, columns)

##### rename columns
data_cleaning.rename(columns={'  num_pages': 'num_pages'}, inplace=True)
##### Converting all columns to numeric
data_cleaning[list(data_cleaning)] = data_cleaning[list(data_cleaning)].astype(float)

path_export = 'data_cleaning.csv'
data_cleaning.to_csv(work+path_export,
               sep=';',
               na_rep='',
               header=True,
               index=False,
               decimal='.',
               encoding='utf-8-sig',
               date_format='%d/%m/%Y')



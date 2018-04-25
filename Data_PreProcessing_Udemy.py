# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 13:06:16 2018

@author: arunabhagupta
"""
#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("E:/AI_ML_DA/R_Python_Code_Datasets/DataSets/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Data.csv")
dataset.head()
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#Taking care of missing data using Scikit Learn
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy="mean",axis=0)
imputer = imputer.fit(X[:,1:3]) # X[:,1:3] here inside bracket, first : means select all rows, then select column index 1 and index 2 as in python index starts from 0
X[:,1:3] = imputer.transform(X[:,1:3])

# Encoding Categorical Data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting the dataset into Training set and Test set
from sklearn.cross_validation import train_test_split #Use  from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

# Feature Scalng
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #We don not need to fit the sc_X object to the test set coz it is already fitted to the Training Set

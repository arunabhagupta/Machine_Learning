# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 00:50:17 2018

@author: aruna
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


#Splitting the dataset into Training set and Test set
from sklearn.cross_validation import train_test_split #Use  from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #We don not need to fit the sc_X object to the test set coz it is already fitted to the Training Set
"""


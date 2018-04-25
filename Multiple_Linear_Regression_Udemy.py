# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 20:26:02 2018

@author: aruna
"""

# Multiple Linear Regression
# Several Independent variables

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('E:/AI_ML_DA/R_Python_Code_Datasets/DataSets/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')
dataset.head()
X = dataset.iloc[:,:-1].values
#X = pd.DataFrame(X)
y = dataset.iloc[:,4].values

# Encoding Categorical Data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
# Python Library for Linear Regression is taking care of the dummy variable trap
# Some libraries we need to take one dummy variable away manually
X = X[:, 1:]

#Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split #Use  from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

# Feature Scaling
# Not required for Multiple Linear Regression. Library will take care of it
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #We don not need to fit the sc_X object to the test set coz it is already fitted to the Training Set
"""

# Fitting Multiple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test Set Results
y_pred = regressor.predict(X_test)

# Building the Optimal model using Backward Elimination
import statsmodels.formula.api as sm
# axis = 0 means adding ones to rows and axis = 1 means adding ones to columns
# the below code will give us ones added to the end of the maxtrix since we are appending it with X
# X = np.append(arr = X, values = np.ones((50,1)).astype(int), axis = 1)
# to add the columns of ones to the beginning of the matrix, we just need to do the below change
# Now x0 = 1
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[: ,  [0,1,2,3,4,5]]
# OLS (Ordinary Least Squares): new library of stats models library 
# Check OSL() help for details about the constant (the intercept)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Step 2 in Backward elimination is done!
regressor_OLS.summary() # Const is the X0 value, X1 and X2 are the dummy variables, X3 is R&D, X4 is Administration, X5 is marketing
# Now removing X2 as X2 has a P-value of 0.990 that is 99%
# Now removing the index 2 as X2 has the highest P-Value
X_opt = X[: ,  [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[: ,  [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[: ,  [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[: ,  [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
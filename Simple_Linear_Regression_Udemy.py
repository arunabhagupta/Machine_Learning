# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 07:05:41 2018

@author: aruna
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("E:/AI_ML_DA/R_Python_Code_Datasets/DataSets/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv")
dataset.head()
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


#Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3,random_state = 0)

# Feature Scalng
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #We don not need to fit the sc_X object to the test set coz it is already fitted to the Training Set
"""

# Fitting simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test set result
# Vector of predicted values
# y_pred will contain all the predicted values for salaries
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

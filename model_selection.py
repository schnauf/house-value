#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 14:16:19 2017

@author: richard
"""

import pandas as pd
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# Location of the training data
data_directory = 'data'
data_file = '{}/housing.csv'.format(data_directory)

# Read dataset and split it into dependent variables (x) and target (y)
housing = pd.read_csv(data_file)
x, y = shuffle(housing.iloc[:,:-1], housing.iloc[:,-1])

# Try a few models in different categories (linear, support vector machine,
# nearest neighbor regression, decision tree regression, ensemble methods)
models = [LinearRegression(), Ridge(), SVR(), KNeighborsRegressor(),
          DecisionTreeRegressor(), RandomForestRegressor(), 
          GradientBoostingRegressor()
          ]


tmp = []
for model in models:
    # Get name of the model as a string
    name = str(model)
    name = name[:name.index('(')]
    # Calculate 10 fold cross validation scores
    scores = cross_val_score(model, x, y, cv = 10, scoring = 'r2')
    # Write data
    tmp.append({'Model_name': name, 'R^2_score': scores.mean()})

TestModels = pd.DataFrame(tmp)

# Print results sorted from best to worse
print 'Results of 10 fold cross validation (from best to worst):'
print TestModels.sort_values(by = 'R^2_score', ascending = False)
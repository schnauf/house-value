#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.externals import joblib

# Location where the final model file will be written
model_directory = 'model'
model_file = '{}/model.pkl'.format(model_directory)

# Location of the training data
data_directory = 'data'
data_file = '{}/housing.csv'.format(data_directory)

# Read dataset and split it into dependent variables (x) and target (y)
housing = pd.read_csv(data_file)
x, y = shuffle(housing.iloc[:,:-1], housing.iloc[:,-1])

# Split data into development and evaluation sets for model training
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

# Select model parameters to be trained
# Test two parameter grids
tuned_parameters = [{'n_estimators': [50, 100, 150, 200], 
                     'max_depth': [3, 4, 5, 6],
                     'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
                     'loss': ['huber', 'quantile']},
                    {'n_estimators': [50, 100, 150, 200], 
                     'max_depth': [3, 4, 5, 6],
                     'loss': ['ls', 'lad']}]

# Setup grid for 10 fold cross validated parameter training
grid = GridSearchCV(GradientBoostingRegressor(), tuned_parameters, cv=10)

# Search the parameter space
grid.fit(X_train, Y_train)

# Print results
print 'Grid R^2 scores on development set:'
print ''

means = grid.cv_results_['mean_test_score']
for mean, params in zip(means, grid.cv_results_['params']):
    print '{:0.3f} for {:s}'.format(mean, params)
print ''

print 'Best parameters found for development set:'
print(grid.best_params_)
print ''

# Final evaluation is done on evaluation set which was not part of the
# model training
print 'Final scores:'
print 'The model is trained on the full development set.'
print 'The scores are computed on the full evaluation set.'
print ''

# Final R^2 score and standard deviation of the estimate
y_true, y_pred = Y_test, grid.predict(X_test)
score = r2_score(y_true, y_pred)
stddev = np.std(y_true - y_pred)

print 'R^2 score: ' + '{:0.3f}'.format(score)
print 'Standard deviation of the estimate: ' + '{:0.3f}'.format(stddev)
print ''

# Construct the final model by fitting to the entire dataset
print 'Fitting final model to entire dataset'
model = GradientBoostingRegressor().set_params(**grid.best_params_)
model.fit(x, y)

# Write final model and standard deviation to file
joblib.dump({'model': model, 'stddev': stddev}, model_file)
print 'Final model written to ' + model_file

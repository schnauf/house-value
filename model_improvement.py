import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from joblib import dump

# Location where the final model file will be written
MODEL_DIRECTORY = "model"
MODEL_FILE = f"{MODEL_DIRECTORY}/model.pkl"

# Location of the training data
DATA_DIRECTORY = "data"
DATA_FILE = f"{DATA_DIRECTORY}/housing.csv"

# Read dataset and split it into dependent variables (x) and target (y)
housing = pd.read_csv(DATA_FILE)
x, y = shuffle(housing.iloc[:, :-1], housing.iloc[:, -1])

# Split data into training and evaluation sets for model training
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

# Select model parameters to be trained
# Test two parameter grids
tuned_parameters = [{"n_estimators": [50, 100, 150, 200],
                     "max_depth": [3, 4, 5, 6],
                     "alpha": [0.1, 0.3, 0.5, 0.7, 0.9],
                     "loss": ["huber", "quantile"]},
                    {"n_estimators": [50, 100, 150, 200],
                     "max_depth": [3, 4, 5, 6],
                     "loss": ["ls", "lad"]}]

# Setup grid for 10 fold cross validated parameter training
grid = GridSearchCV(GradientBoostingRegressor(), tuned_parameters, cv=10)

# Search the parameter space
grid.fit(X_train, Y_train)

# Put results into a dataframe and sort by R^2 score
model_scores = pd.DataFrame(grid.cv_results_['params'])
model_scores["mean_test_score"] = grid.cv_results_["mean_test_score"]
model_scores.sort_values(by="mean_test_score", ascending=False, inplace=True)

# Print results sorted from best to worst
print("Grid R^2 scores on training set:")
print(model_scores.to_string(index=False))
print("\n")

print("Best parameters found for training set:")
print(grid.best_params_)
print("\n")

# Final R^2 score and standard deviation of the estimate
# Final evaluation is done on evaluation set which was not part of the
# model training
y_true, y_pred = Y_test, grid.predict(X_test)
score = r2_score(y_true, y_pred)
stddev = np.std(y_true - y_pred)


print("Final scores:")
print(f"R^2 score: {score:0.3f}")
print(f"Standard deviation of the estimate: {stddev:0.3f}")
print("The model is trained on the full training set.")
print("The scores are computed on the full evaluation set.")
print("\n")

# Construct the final model by fitting to the entire dataset
print("Fitting final model to entire dataset:")
model = GradientBoostingRegressor().set_params(**grid.best_params_)
model.fit(x, y)

# Write final model and standard deviation to file
dump({"model": model, "stddev": stddev}, MODEL_FILE)
print(f"Final model written to {MODEL_FILE}")

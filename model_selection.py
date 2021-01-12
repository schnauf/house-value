import pandas as pd
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# Location of the training data
DATA_DIRECTORY = "data"
DATA_FILE = f"{DATA_DIRECTORY}/housing.csv"

# Read dataset and split it into dependent variables (x) and target (y)
housing = pd.read_csv(DATA_FILE)
x, y = shuffle(housing.iloc[:, :-1], housing.iloc[:, -1])

# Try a few models in different categories (linear, support vector machine,
# nearest neighbor regression, decision tree regression, ensemble methods)
models = [LinearRegression(), Ridge(), SVR(), KNeighborsRegressor(),
          DecisionTreeRegressor(), RandomForestRegressor(),
          GradientBoostingRegressor()]


# Calculate 10 fold cross validation scores of the models
results = []
for model in models:
    NAME = type(model).__name__
    scores = cross_val_score(model, x, y, cv=10, scoring='r2')
    results.append({"model_name": NAME, "mean_test_score": scores.mean()})

# Put results into a dataframe and sort by R^2 score
model_scores = pd.DataFrame(results)
model_scores.sort_values(by="mean_test_score", ascending=False, inplace=True)

# Print results sorted from best to worst
print("Results of 10 fold cross validation (from best to worst):\n")
print(model_scores.to_string(index=False))

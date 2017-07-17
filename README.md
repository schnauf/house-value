# Flask API for scikit learn
A simple Flask application that implements a regression model to predict house values based on training data.  Reads a pickled sklearn model into memory when the Flask app is started and returns predictions through the /predict endpoint. Any sklearn model can be used for prediction.

### Dependencies
- scikit-learn
- Flask
- pandas
- numpy

```
pip install -r requirements.txt
```

# Model
At the selection stage, a few different regression models are tested. At the improvement stage, the best model (gradient boosting regression) is further tuned to build the final regression model, which is written to file.

# Endpoints
### /predict (POST)
When given a JSON object representing independent variables, the program returns a JSON object representing the predicted house value and the standard deviation of the estimate. Example request (assuming that the server is running on port 5000 on localhost):
```

curl http://localhost:5000/predict -H "Content-Type: application/json" --data-binary '{
  "crime_rate": 0.1,
  "avg_number_of_rooms": 4.0,
  "distance_to_employment_centers": 6.5,
  "property_tax_rate": 330.0,
  "pupil_teacher_ratio": 19.5
}'
```

and sample output (precise values will depend on the regression model):
```

{
  "house_value": 18.8, 
  "stddev": 3.9
}
```

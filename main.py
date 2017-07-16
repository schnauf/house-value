# -*- coding: utf-8 -*-


from sklearn.externals import joblib
from flask import Flask, abort, request, jsonify, make_response
app = Flask(__name__)

# Load fitted model
model_params = joblib.load('model/model.pkl')
model = model_params['model']
stddev = model_params['stddev']

@app.route('/predict', methods = ['POST'])
def predict():
    # Get request parameters
    content = request.get_json()
    
    # Read input values
    try:
        crime_rate = content['crime_rate']
        avg_number_of_rooms = content['avg_number_of_rooms']
        distance_to_employment_centers = content['distance_to_employment_centers']
        property_tax_rate = content['property_tax_rate']
        pupil_teacher_ratio = content['pupil_teacher_ratio']

        # Test vector
        Xtest = [[crime_rate, avg_number_of_rooms,
                  distance_to_employment_centers,
                  property_tax_rate, pupil_teacher_ratio]]
    
        # Calculate house value based on input parameters
        house_value = model.predict(Xtest)[0]
    
    except (KeyError, TypeError, ValueError):
        abort(400)
    
    # Format response and return it
    response = {'house_value': round(house_value, 1), 
                'stddev': round(stddev, 1)}
    return jsonify(response)

# Retun 400 Bad request error if required parameters are missing
# or request is not properly formatted
@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad request'}), 400)

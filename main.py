from joblib import load
from flask import Flask, abort, request, jsonify, make_response
app = Flask(__name__)

# Load fitted model
model_params = load('model/model.pkl')
model = model_params['model']
stddev = model_params['stddev']


def get_input_data(input_data):
    """Get input data from HTTP request.

    Parameters:
        input_data (flask request object): Request containing input JSON

    Returns:
        request_dat (list): Input data points
    """
    # Get input json
    if input_data.is_json:
        content = input_data.get_json()
    else:
        abort(400, "Input data not in proper JSON format")

    # Read input values
    try:
        crime_rate = content['crime_rate']
        avg_number_of_rooms = content['avg_number_of_rooms']
        distance_to_employment_centers = content['distance_to_employment_centers']
        property_tax_rate = content['property_tax_rate']
        pupil_teacher_ratio = content['pupil_teacher_ratio']

        # Test vector
        request_data = [[crime_rate, avg_number_of_rooms,
                        distance_to_employment_centers,
                        property_tax_rate, pupil_teacher_ratio]]

        return request_data

    except KeyError:
        abort(400, "Input data missing")


@app.route('/predict', methods=['POST'])
def predict():
    """Predict house value from input data based on fitted model.

    Returns:
        response (JSON object): House value and model standard deviation
    """
    # Get input data points from HTTP request
    x_test = get_input_data(request)

    try:
        # Calculate house value based on fitted model
        house_value = model.predict(x_test)[0]

        # Format response and return it
        response = {'house_value': round(house_value, 1),
                    'stddev': round(stddev, 1)}
        return jsonify(response)

    except ValueError:
        abort(400, "Bad input value")


# Retun 400 Bad request error if required parameters are missing
# or request is not properly formatted
@app.errorhandler(400)
def bad_request(error):
    """Return error message."""
    return make_response(jsonify({'error': error.description}), 400)

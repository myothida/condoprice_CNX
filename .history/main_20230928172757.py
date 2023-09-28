from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
pipe = pickle.load(open('CNX.pkl', 'rb'))

# Define the index route
@app.route('/')
def index():
    return render_template('index.html')

# Define the predict route to handle form submissions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Extract user input from the form
            sqm = (request.form.get('sqm'))  # Ensure sqm is converted to float
            bedroom = (request.form.get('bedroom'))  # Ensure bedroom is converted to int
            bathroom = (request.form.get('bathroom'))  # Ensure bathroom is converted to int
            parking = (request.form.get('parking'))  # Ensure parking is converted to int

            # Create a DataFrame with the correct order of features
            input_data = pd.DataFrame([[sqm, bedroom, bathroom, parking]], columns=['sqm', 'bedroom', 'bathroom', 'parking'])

            # Debugging: Print the input_data
            print("Input Data:")
            print(input_data)

            # Make the prediction using the model
            prediction = pipe.predict(input_data)[0] * 1e5

            # Return the prediction result as JSON
            return jsonify({'prediction': np.round(prediction, 2)})
    except Exception as e:
        # Log the error for debugging
        print("Error:", str(e))
        return jsonify({'error': 'An error occurred'}), 500  # Return an error response with status code 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5001)

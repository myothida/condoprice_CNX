from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


app = Flask(__name__)

# Load the trained model
with open('data.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the index route
@app.route('/')
def index():
    return render_template('index.html')

# Modify your predict route
# Modify your predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract user input from the JSON data sent by the front-end
        data = request.get_json()

        # Extract values from the data
        sqm = float(data['sqm'])
        bedroom = int(data['bedroom'])
        bathroom = int(data['bathroom'])
        parking = int(data['parking'])

        # Create a DataFrame with the correct order of features
        input_data = pd.DataFrame([[1, sqm, bedroom, bathroom, parking]], columns=['1', 'sqm', 'bedroom', 'bathroom', 'parking'])

        # Define the same preprocessing steps as in training
        scaler = StandardScaler()  # Define the scaler
        polytransform = PolynomialFeatures(degree=2)  # Define the PolynomialFeatures transformer

        # Fit the scaler and transformer on training data (you can load training data if needed)
        scaler.fit(X_train)
        polytransform.fit(X_train)

        # Perform the same preprocessing steps as in training
        input_data_scaled = scaler.transform(input_data)
        input_data_poly = polytransform.transform(input_data_scaled)

        # Make sure to load your model as well, similar to how you did it in training
        with open('data.pkl', 'rb') as file:
            model = pickle.load(file)

        # Make the prediction using the model
        prediction = model.predict(input_data_poly)[0] * 1e5

        # Return the prediction result as JSON
        return jsonify({'prediction': np.round(prediction, 2)})


    
# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5001)

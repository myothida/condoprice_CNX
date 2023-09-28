from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
pipe = pickle.load(open('data.pkl', 'rb'))

# Define the index route
@app.route('/')
def index():
    return render_template('index.html')

# Define the predict route to handle form submissions
@app.route('/predict', methods=['POST'])
def predict():
    
    ['1' 'sqm' 'bedroom' 'bathroom' 'parking' 'sqm^2' 'sqm bedroom'
    'sqm bathroom' 'sqm parking' 'bedroom^2' 'bedroom bathroom'
    'bedroom parking' 'bathroom^2' 'bathroom parking' 'parking^2']


    if request.method == 'POST':
        # Extract user input from the JSON data sent by the front-end
        data = request.get_json()

        # Extract values from the data
        sqm = float(data['sqm'])
        bedroom = int(data['bedroom'])
        bathroom = int(data['bathroom'])
        parking = int(data['parking'])

        # Create a DataFrame with the correct order of features
        input_data = pd.DataFrame([[sqm, bedroom, bathroom, parking]], columns=['sqm', 'bedroom', 'bathroom', 'parking'])

        # Print debugging information
        print("Input Data:")
        print(input_data)
        
        feature_names = pipe.named_steps['polytransform'].get_feature_names_out(input_data.columns)
        print("Feature Names in the Model:")
        print(feature_names)

        # Make the prediction using the model
        prediction = pipe.predict(input_data)[0] * 1e5

        # Return the prediction result as JSON
        return jsonify({'prediction': np.round(prediction, 2)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5001)

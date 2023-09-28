from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
pipe = pickle.load(open('model.pkl', 'rb'))

# Define the index route
@app.route('/')

def index():
    return render_template('index.html')

# Define the predict route to handle form submissions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract user input from the form
        price_per_sqm = int(request.form.get('price_per_sqm'))
        sqm = int(request.form.get('sqm'))
        bedroom = int(request.form.get('bedroom'))
        bathroom = int(request.form.get('bathroom'))
        parking = int(request.form.get('parking'))

        # Create a DataFrame with the correct order of features
    input_data = pd.DataFrame([[sqm, bedroom, bathroom, parking]], columns=['sqm', 'bedroom', 'bathroom', 'parking'])

        # Make the prediction using the model
    prediction = pipe.predict(input_data)[0] * 1e5

        # Return the prediction result as JSON
    return jsonify({'prediction': np.round(prediction, 2)})
   



# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5001)




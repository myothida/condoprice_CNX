from flask import Flask, render_template, request 
import pandas as pd
import pickle
import numpy as np


app = Flask(__name__)
data = pd.read_csv('data.csv')
pipe = pickle.load(open('BestModel_CNX.pkl','rb'))


@app.route('/')
def index():

    return render_template('index.html' )


@app.route('/predict', methods=['POST'])
def predict():
    sqm = float(request.form.get('sqm'))
    bedroom = request.form.get('bedroom')
    bathroom = int(request.form.get('bathroom'))
    parking = int(request.form.get('parking'))

    # Handle the 'Studio' option as a special case
    if bedroom == 'Studio':
        bedroom_value = 0
    else:
        bedroom_value = int(bedroom)

    # Create a DataFrame with the correct order of features
    input_data = pd.DataFrame([[sqm, bedroom_value, bathroom, parking]], columns=['sqm', 'bedroom', 'bathroom', 'parking'])

    # Make the prediction using the model
    prediction = pipe.predict(input_data)[0] * 1e5

    return str(np.round(prediction, 2))



if __name__ == "__main__":
    app.run(debug=True, port=5001)

from flask import Flask, render_template, request 
import pandas as pd
import pickle
import numpy as np


app = Flask(__name__)
data = pd.read_csv('data.csv')
pipe = pickle.load(open('BestModel_CNX.pkl','rb'))



# Load the pickled model from the file
with open('BestModel_CNX.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Now, you can inspect the model and its attributes
# For example, you can print the feature names or model parameters
print("Feature Names:", loaded_model.feature_names)
print("Model Parameters:", loaded_model.get_params())

@app.route('/')
def index():

    return render_template('index.html' )




@app.route('/predict', methods=['POST'])
def predict():
    sqm = float(request.form.get('sqm'))
    bedroom = int(request.form.get('bedroom'))
    bathroom = int(request.form.get('bathroom'))
    parking = int(request.form.get('parking'))

    # Create a DataFrame with the correct order of features
    input_data = pd.DataFrame([[sqm, bedroom, bathroom, parking]], columns=['sqm', 'bedroom', 'bathroom', 'parking'])

    # Make the prediction using the model
    prediction = pipe.predict(input_data)[0] * 1e5

    return str(np.round(prediction, 2))



if __name__ == "__main__":
    app.run(debug=True, port=5001)



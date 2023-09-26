from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('Data_midterm.csv')
pipe = pickle.load(open('BestModel_CNX.pkl','rb'))


@app.route('/')
def index():
    
    
    locations = sorted(data['location'].unique()) 
    return render_template('index.html' ,locations=locations)


@app.route('/predict' , methods=['POST'])
def predict():
    
    
    locations = request.form.get('location')
    sqm = request.form.get('sqm')
    bedroom = request.form.get('bedroom')
    bathroom = request.form.get('bathroom')
    parking = request.form.get('parking')
    
    print(locations ,sqm , bedroom , bathroom , parking )
    input = pd.DataFrame([[locations,sqm,bedroom,bathroom,parking]],columns=['location' ,'sqm' ,'bedroom','bathroom','parking'])
    prediction = pipe.predict(input)
    
    return str(prediction)


if __name__ == "__main__":
    app.run(debug=True, port=5001)

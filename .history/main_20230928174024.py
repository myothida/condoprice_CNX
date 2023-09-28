from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

# import lib
import pandas as pd
import pickle

# Read data 
df = pd.read_csv('data.csv')

# see all location name 
locations = df['location'].unique()
print(locations)


#set x and y 
y = df['price'] 
X = df.drop(['price', 'location'], axis = 'columns')



#Show count of records
num_records = df.shape[0]

print("Number of Records:", num_records)

#Count of locations
location_counts = df['location'].value_counts()
print(location_counts)





from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

X_train

from sklearn.preprocessing import StandardScaler, PolynomialFeatures , MinMaxScaler
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Set regressors to find r2 mae 
regressors = {
    'Linear Regression': LinearRegression(),

    'Ridge Regression': Ridge(alpha=1000.0,  
                              fit_intercept=True,   
                              solver='auto',      
                              max_iter=None 
                              ),
    
    'Lasso Regression': Lasso(alpha=10000.0,         
                              fit_intercept=True,   
                              precompute=False,   
                              max_iter=1000,      
                              warm_start=False,   
                              positive=False,     
                              selection='cyclic'
                              ),
    
    'Random Forest Regression': RandomForestRegressor(n_estimators=500, max_depth=20, 
                                                      min_samples_split=5, min_samples_leaf=1, 
                                                      bootstrap=True, random_state=42)
}

# use pipeline model
pipelines = {}
for name, regressor in regressors.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('transform', PolynomialFeatures(degree=2)),
        ('regressor', regressor)
    ])
    pipelines[name] = pipeline

results = {}
for name, pipeline in pipelines.items():
    
    model = pipeline.fit(X_train, y_train)
    
    #Set result and input training
    y_pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train,y_pred)
    
    #Set result and input testing
    y_predt = model.predict(X_test)
    maet = mean_absolute_error(y_test, y_predt)
    r2t = r2_score(y_test,y_predt)
    
    results[name] = [mae, r2, maet, r2t] 

df_result = pd.DataFrame(results)
df_result = df_result.round(2)
df_result.index = ['mae_train', 'r2_train', 'mae_test', 'r2_test']

#Print result
df_result


steps = [("imp_mean", SimpleImputer()), ("scale", StandardScaler()), 
         ("polytransform", PolynomialFeatures(degree =2)), ("regressor", RandomForestRegressor()) ]

pipeline = Pipeline(steps)
model = pipeline.fit(X_train, y_train)






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

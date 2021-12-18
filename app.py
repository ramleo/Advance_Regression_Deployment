import werkzeug
import numpy as np
import pandas as pd
from joblib import load
from flask import Flask, request, jsonify, render_template

# Creating app object

app = Flask(__name__)

# Providing min and max range as this is linear regression model
# this model cannot extrapolate hence values provided must be
# within min and max range

values_range = {
 'ExterQual':{'min': 0, 'max': 3},
 'KitchenQual': {'min': 0, 'max': 3},
 'BsmtQual': {'min': 0, 'max': 4},
 'GarageFinish': {'min': 0, 'max': 3},
 'OverallCond': {'min': 1, 'max': 10},
 'TotalBsmtSF': {'min': 0, 'max': 6110},
 '1stFlrSF': {'min': 334, 'max': 4692},
 'GrLivArea': {'min': 334, 'max': 5642},
 'FullBath': {'min': 0, 'max': 3},
 'GarageCars': {'min': 0, 'max': 4},
 'GarageArea': {'min': 0, 'max': 1418}
}

## Predicting Test data## Checking the model with single data point from test dataset# Loading required models

# Loading the saved model
model = load('udf_model.joblib')
# Standard scaling the variables
sc = load('udf_standscaler.joblib')

# Using the html template
@app.route('/')
def home():
    return render_template('index.html')

# Exposing the below code to localhost:5000
@app.route('/api', methods=['POST'])
def pred_testquery():

    # content = request.json
    ExterQual = request.form['ExterQual']
    KitchenQual = request.form['KitchenQual']
    BsmtQual = request.form['BsmtQual']
    GarageFinish = request.form['GarageFinish']
    OverallCond = request.form['OverallCond']
    TotalBsmtSF = request.form['TotalBsmtSF']
    FirstFlrSF = request.form['1stFlrSF']
    GrLivArea = request.form['GrLivArea']
    FullBath = request.form['FullBath']
    GarageCars = request.form['GarageCars']
    GarageArea = request.form['GarageArea']

    datapoint = [[ExterQual, KitchenQual, BsmtQual, GarageFinish, OverallCond,
                  TotalBsmtSF, FirstFlrSF, GrLivArea, FullBath, GarageCars, GarageArea]]

    datadict = {'ExterQual':int(ExterQual), 'KitchenQual':int(KitchenQual),'BsmtQual':int(BsmtQual),
                'GarageFinish':int(GarageFinish),'OverallCond':int(OverallCond),'TotalBsmtSF':int(TotalBsmtSF),
                '1stFlrSF':int(FirstFlrSF),'GrLivArea':int(GrLivArea),
                'FullBath':int(FullBath),'GarageCars':int(GarageCars),'GarageArea':int(GarageArea)}
    errors = []

    for i in datadict:
        if i in values_range:
            min_range = values_range[i]['min']
            max_range = values_range[i]['max']
            value = datadict[i]
            if value < min_range or value > max_range:
                errors.append(f'The values should be between {min_range} and {max_range}.')
        else:
            errors.append(f'Unexpected field: {i}.')
    #
    for i in values_range:
        if i not in datadict:
            errors.append(f'Missing value: {i}')
    #
    if len(errors) < 1:

        def y_pred(X_data=datapoint, stdscale=sc, model=model):

            # Transform test data
            X_data = stdscale.transform(X_data)

            # Fitting Linear Regression model
            pred = model.predict(X_data)

            return pred

        prediction = y_pred()
        Sale_Price = float(prediction)
        response = {'Sale Price is': np.round(Sale_Price,3), 'errors': errors}

    else:
        response = {'errors': errors}

    return str(response)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
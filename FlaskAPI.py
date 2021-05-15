# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 03:35:38 2020

@author: Student
"""

from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
import sys
# Your API definition
app = Flask(__name__)

path = 'C:/Jatin/code/MachineLearning'



# Input 
# ['MonthlyCharges',
#  'TotalCharges',
#  'gender_Female',
#  'gender_Male',
#  'SeniorCitizen_0',
#  'SeniorCitizen_1',
#  'Partner_No',
#  'Partner_Yes',
#  'Dependents_No',
#  'Dependents_Yes',
#  'PhoneService_No',
#  'PhoneService_Yes',
#  'MultipleLines_No',
#  'MultipleLines_No phone service',
#  'MultipleLines_Yes',
#  'InternetService_DSL',
#  'InternetService_Fiber optic',
#  'InternetService_No',
#  'OnlineSecurity_No',
#  'OnlineSecurity_Yes',
#  'OnlineBackup_No',
#  'OnlineBackup_Yes',
#  'DeviceProtection_No',
#  'DeviceProtection_Yes',
#  'TechSupport_No',
#  'TechSupport_Yes',
#  'StreamingTV_No',
#  'StreamingTV_Yes',
#  'StreamingMovies_No',
#  'StreamingMovies_Yes',
#  'Contract_Month-to-month',
#  'Contract_One year',
#  'Contract_Two year',
#  'PaperlessBilling_No',
#  'PaperlessBilling_Yes',
#  'PaymentMethod_Bank transfer (automatic)',
#  'PaymentMethod_Credit card (automatic)',
#  'PaymentMethod_Electronic check',
#  'PaymentMethod_Mailed check',
#  'Tenure_Group_0_1_year',
#  'Tenure_Group_1_2_year',
#  'Tenure_Group_2_3_year',
#  'Tenure_Group_3_4_year',
#  'Tenure_Group_4_5_year']


@app.route("/", methods=['GET'])
def indexMethod():
    return "go to /predict"

@app.route("/predict", methods=['GET','POST']) #use decorator pattern for the route
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(lr.predict(query))
            print({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction)})
            return "Welcome to titanic model APIs!"

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load(path + '/model_DecisionTree.pkl') # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load(path + '/model_columns_DecisionTree.pkl') # Load "model_columns.pkl"
    print ('Model columns loaded')
    
    app.run(port=port, debug=True)

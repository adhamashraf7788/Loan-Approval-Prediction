from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained models
with open('models/svm.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('models/Knn.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('models/logistic.pkl', 'rb') as f:
    logistic_model = pickle.load(f)

# Define one-hot encoding mappings
mappings = {
    'Married': {'No': 0, 'Yes': 1},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education': {'Graduate': 1, 'Not Graduate': 0}
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = {
        'Married': request.form['married'],
        'Dependents': int(request.form['dependents']),
        'Education': request.form['education'],
        'Self_Employed': request.form['self_employed'],
        'ApplicantIncome': float(request.form['applicant_income']),
        'CoapplicantIncome': float(request.form['coapplicant_income']),
        'LoanAmount': float(request.form['loan_amount']),
        'Loan_Amount_Term': float(request.form['loan_amount_term']),
        'Credit_History': float(request.form['credit_history']),
        'Property_Area': request.form['property_area']
    }

    # Create DataFrame
    df = pd.DataFrame([data])

    # Apply one-hot encoding
    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)

    # Prepare features for prediction
    features = df[['Married', 'Dependents', 'Education', 'Self_Employed', 
                   'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                   'Loan_Amount_Term', 'Credit_History', 'Property_Area']]

    # Make predictions
    svm_pred = svm_model.predict(features)[0]
    rf_pred = rf_model.predict(features)[0]
    logistic_pred = logistic_model.predict(features)[0]

    # Convert predictions to readable format
    pred_map = {0: 'Not Approved', 1: 'Approved'}
    predictions = {
        'SVM': pred_map[svm_pred],
        'Knn': pred_map[rf_pred],
        'Logistic Regression': pred_map[logistic_pred]
    }

    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
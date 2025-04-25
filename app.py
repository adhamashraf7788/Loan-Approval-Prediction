from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load all models
logistic_model = joblib.load(os.path.join('models', 'C:\Users\co.magic\Desktop\ML\ML-Project\models\logistic_model.pkl'))
svm_model = joblib.load(os.path.join('models', 'C:\Users\co.magic\Desktop\ML\ML-Project\models\svm_model'))
rf_model = joblib.load(os.path.join('models', 'C:\Users\co.magic\Desktop\ML\ML-Project\models\random_forest_model'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.getlist('features')]
    model_choice = request.form['model']

    if model_choice == 'logistic':
        model = logistic_model
    elif model_choice == 'svm':
        model = svm_model
    else:
        model = rf_model

    prediction = model.predict([data])
    result = "Loan will default" if prediction[0] == 1 else "Loan will not default"
    return render_template('index.html', result=result)

if __name__ == 'main':
    app.run(debug=True)
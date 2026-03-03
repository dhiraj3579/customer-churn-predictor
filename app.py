from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    # Show the website frontend
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the website form
    tenure = float(request.form['tenure'])
    monthly_charges = float(request.form['monthly_charges'])
    support_calls = int(request.form['support_calls'])

    # Format the data for the model
    features = np.array([[tenure, monthly_charges, support_calls]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Format the result
    if prediction == 1:
        result = "High Risk of Churn! 🚨"
    else:
        result = "Customer is likely to stay. ✅"

    # Send the result back to the website
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)
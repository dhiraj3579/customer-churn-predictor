import os
import pickle
import pandas as pd
import io
from flask import Flask, render_template, request, Response

app = Flask(__name__, template_folder='templates')

# Initialize history to store session data
predictions_history = []

# Load the trained model
try:
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html', history=predictions_history)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file was uploaded
        if 'file_upload' in request.files and request.files['file_upload'].filename != '':
            file = request.files['file_upload']
            df_upload = pd.read_csv(file)
            
            # CLEANING: Make sure columns match your model (Tenure, Monthly_Charges, Support_Calls)
            # This fixes small typos like 'tenure' or 'Monthly Charges'
            df_upload.columns = df_upload.columns.str.strip().str.replace(' ', '_').str.title()

            # EXTRA COLUMN PROTECTION: Only select the 3 columns the model knows
            X_bulk = df_upload[['Tenure', 'Monthly_Charges', 'Support_Calls']]
            
            # Make bulk predictions
            preds = model.predict(X_bulk)
            df_upload['Prediction'] = ["High Risk" if p == 1 else "Low Risk" for p in preds]
            
            # Add results to our history list
            new_records = df_upload.to_dict(orient='records')
            predictions_history.extend(new_records)
            
            return render_template('index.html', 
                                 prediction_text=f"Success! Processed {len(preds)} customers from CSV.", 
                                 history=predictions_history)

        else:
            # Standard Manual Input logic
            tenure = float(request.form.get('tenure', 0))
            monthly_charges = float(request.form.get('monthly_charges', 0))
            support_calls = float(request.form.get('support_calls', 0))

            input_data = pd.DataFrame([[tenure, monthly_charges, support_calls]], 
                                      columns=['Tenure', 'Monthly_Charges', 'Support_Calls'])

            prediction = model.predict(input_data)[0]
            result = "High Risk of Churn" if prediction == 1 else "Low Risk (Stay)"
            
            predictions_history.append({
                'Tenure': tenure, 'Monthly_Charges': monthly_charges, 
                'Support_Calls': support_calls, 'Prediction': result
            })
            
            return render_template('index.html', prediction_text=result, history=predictions_history)
            
    except Exception as e:
        return f"Error: {str(e)}. Please ensure CSV has columns: Tenure, Monthly_Charges, Support_Calls"

@app.route('/download')
def download():
    if not predictions_history:
        return "No data available", 400
    df = pd.DataFrame(predictions

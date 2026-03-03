import os
import pickle
import pandas as pd
import io
from flask import Flask, render_template, request, Response

app = Flask(__name__, template_folder='templates')

# Global list to store session predictions (simple version)
# Note: In a real app, you'd use a database like SQLite
predictions_history = []

with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tenure = float(request.form['tenure'])
    monthly_charges = float(request.form['monthly_charges'])
    support_calls = float(request.form['support_calls'])

    input_data = pd.DataFrame([[tenure, monthly_charges, support_calls]], 
                              columns=['tenure', 'monthly_charges', 'support_calls'])

    prediction = model.predict(input_data)[0]
    result = "High Risk of Churn" if prediction == 1 else "Low Risk (Stay)"
    
    # Save to history
    predictions_history.append({
        'Tenure': tenure,
        'Monthly_Charges': monthly_charges,
        'Support_Calls': support_calls,
        'Prediction': result
    })
    
    return render_template('index.html', prediction_text=result, history=predictions_history)

@app.route('/download')
def download():
    if not predictions_history:
        return "No data to download", 400
    
    # Convert history list to CSV
    df = pd.DataFrame(predictions_history)
    output = io.StringIO()
    df.to_csv(output, index=False)
    
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=churn_predictions.csv"}
    )

if __name__ == "__main__":
    # Render requires the app to listen on 0.0.0.0 and a specific port
    port = int(os.environ.get("PORT", 10000)) 
    app.run(host='0.0.0.0', port=port)

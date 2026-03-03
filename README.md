📊 AI Customer Churn Predictor
An end-to-end Machine Learning web application designed to predict whether a customer is at high risk of canceling their subscription (churning) based on their account usage metrics.

This project demonstrates the full data science lifecycle: from generating data and training a predictive model, to deploying that model as a REST API with a user-friendly frontend.

🚀 Features
Machine Learning Model: Utilizes a Random Forest Classifier trained on customer metrics.

Interactive Web Interface: A clean, responsive frontend built with HTML and Bootstrap.

Real-Time Predictions: Users can input customer data and instantly receive a churn risk assessment.

REST API Backend: Built with Flask to seamlessly connect the frontend to the Scikit-Learn model.

🛠️ Tech Stack
Language: Python

Machine Learning: Scikit-Learn, Pandas, NumPy

Backend: Flask

Frontend: HTML5, CSS3, Bootstrap 5

💻 How to Run Locally
If you would like to run this project on your own machine, follow these steps:

1. Clone the repository

git clone https://github.com/dhiraj3579/customer-churn-predictor.git
cd customer-churn-predictor

2. Install dependencies

pip install -r requirements.txt

3. Train the model

python train_model.py

4. Start the Flask server

python app.py

5. Open the app
Open your web browser and navigate to http://127.0.0.1:5000

🔮 Future Improvements
Swap the synthetic dataset for a real-world dataset (e.g., the Telco Customer Churn dataset from Kaggle).

Add a data visualization dashboard using Matplotlib or Plotly to show churn trends.

Implement a database (SQLite/PostgreSQL) to store user predictions over time.

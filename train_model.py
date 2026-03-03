
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Creating a dummy dataset (In a real project, load a CSV here)
# Features: Tenure (months), Monthly_Charges ($), Support_Calls
data = {
    'Tenure': np.random.randint(1, 72, 1000),
    'Monthly_Charges': np.random.uniform(20.0, 120.0, 1000),
    'Support_Calls': np.random.randint(0, 10, 1000)
}
df = pd.DataFrame(data)

# Target: 1 if they churned (left), 0 if they stayed
# Let's make a rule: High charges + low tenure + high support calls = high chance of churn
df['Churn'] = ((df['Monthly_Charges'] > 80) & (df['Tenure'] < 12) | (df['Support_Calls'] > 5)).astype(int)

# 2. Train the Model
X = df[['Tenure', 'Monthly_Charges', 'Support_Calls']]
y = df['Churn']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. Save the Model to your folder
with open('churn_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as churn_model.pkl!")
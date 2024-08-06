import joblib
import pandas as pd
import json
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import numpy as np

# List of model files
model_files = [
    'best_LogisticRegression_model.joblib',
    'best_SVM_model.joblib',
    'best_KNN_model.joblib',
    'best_RandomForest_model.joblib',
    'best_DecisionTree_model.joblib',
    'best_NaiveBayes_model.joblib',
    'best_XGBoost_model.joblib'
]

# Load the unbalanced dataset
unbalanced_data = pd.read_csv('balanced_dataset.csv')

# Separate features and target
X = unbalanced_data.drop(columns=['HeartDiseaseorAttack'])
y = unbalanced_data['HeartDiseaseorAttack']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection
selector = SelectKBest(f_classif, k=10)  # Select top 10 features
X_selected = selector.fit_transform(X_scaled, y)

# Function to make predictions and get metrics
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    return {
        'accuracy': accuracy,
        'classification_report': report
    }

# Dictionary to store results
unbalanced_results = {}

# Evaluate each model
for model_file in model_files:
    model_name = model_file.split('_')[1]  # Extract model name from file name
    print(f"Evaluating {model_name}...")
    
    # Load the model
    model = joblib.load(model_file)
    
    # Evaluate the model
    results = evaluate_model(model, X_selected, y)
    
    # Store the results
    unbalanced_results[model_name] = results

with open('unbalanced_dataset_results.json', 'w') as f:
    json.dump(unbalanced_results, f, indent=2)

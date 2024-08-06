import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import xgboost as xgb
import json

# Load dataset
data = pd.read_csv('balanced_dataset.csv')

# Separate features and target
X = data.drop(columns=['HeartDiseaseorAttack'])
y = data['HeartDiseaseorAttack']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection
selector = SelectKBest(f_classif, k=10)  # Select top 10 features
X_selected = selector.fit_transform(X_scaled, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

def train_evaluate_save(model, param_grid, name):
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # For json output
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'test_accuracy': accuracy,
        'classification_report': report,
        'cv_results': []
    }
    
    # Add cross-validation results
    for i, (params, mean_score) in enumerate(zip(grid_search.cv_results_['params'],
                                                 grid_search.cv_results_['mean_test_score'])):
        cv_result = {
            'params': params,
            'mean_cv_score': mean_score,
        }
        
        # Try to get individual CV scores if available
        try:
            cv_result['cv_scores'] = grid_search.cv_results_[f'split{i}_test_score'].tolist()
        except KeyError:
            cv_scores = cross_val_score(model.set_params(**params), X_train, y_train, cv=5)
            cv_result['cv_scores'] = cv_scores.tolist()
        
        results['cv_results'].append(cv_result)
    
    with open(f'{name}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save the best model
    joblib.dump(best_model, f'best_{name}_model.joblib')
    
    return best_model, accuracy

models = {
    # Logistic Regression: A linear model for classification that estimates probabilities using a logistic function.
    # Parameters:
    # - C: Inverse of regularization strength. Smaller values specify stronger regularization.
    # - solver: Algorithm to use in the optimization problem. 'lbfgs' is a good default, 'liblinear' is better for small datasets.
    'LogisticRegression': (LogisticRegression(random_state=42, max_iter=10000),
                           {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}),

    # Support Vector Machine (SVM): Finds the hyperplane that best separates classes in a high-dimensional space.
    # Parameters:
    # - C: Regularization parameter. Trades off correct classification of training examples against maximization of the decision function's margin.
    # - kernel: Specifies the kernel type to be used in the algorithm. It maps inputs to higher-dimensional spaces.
    # - degree: Degree of the polynomial kernel function ('poly'). Ignored by other kernels.
    'SVM': (SVC(random_state=42),
            {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly'], 'degree': [2, 3]}),

    # K-Nearest Neighbors (KNN): Classifies based on the k nearest neighbors of each query point.
    # Parameters:
    # - n_neighbors: Number of neighbors to use for k-neighbors queries.
    # - weights: Weight function used in prediction. 'uniform' (default): uniform weights, 'distance': weight points by the inverse of their distance.
    'KNN': (KNeighborsClassifier(),
            {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}),

    # Random Forest: An ensemble of decision trees, where each tree is built on a random subset of features and samples.
    # Parameters:
    # - n_estimators: The number of trees in the forest.
    # - max_depth: The maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contain less than min_samples_split samples.
    'RandomForest': (RandomForestClassifier(random_state=42),
                     {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30]}),

    # Decision Tree: A non-parametric supervised learning method used for classification and regression.
    # Parameters:
    # - max_depth: The maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contain less than min_samples_split samples.
    # - min_samples_split: The minimum number of samples required to split an internal node.
    'DecisionTree': (DecisionTreeClassifier(random_state=42),
                     {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}),

    # Naive Bayes: A probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between the features.
    # Parameters:
    # - var_smoothing: Portion of the largest variance of all features that is added to variances for calculation stability.
    'NaiveBayes': (GaussianNB(),
                   {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}),

    # XGBoost: An optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.
    # Parameters:
    # - n_estimators: Number of gradient boosted trees. Equivalent to number of boosting rounds.
    # - max_depth: Maximum tree depth for base learners.
    # - learning_rate: Boosting learning rate (xgb's "eta").
    'XGBoost': (xgb.XGBClassifier(random_state=42),
                {'n_estimators': [100, 200, 300], 'max_depth': [3, 4, 5], 'learning_rate': [0.01, 0.1, 0.3]})
}

# Train, evaluate, and save results for each model
results = {}
for name, (model, param_grid) in models.items():
    print(f"\nTraining {name}...")
    best_model, accuracy = train_evaluate_save(model, param_grid, name)
    results[name] = accuracy

# Print overall results
print("\nOverall Results:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.4f}")
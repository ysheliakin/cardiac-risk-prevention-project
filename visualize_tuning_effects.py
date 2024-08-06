import json
import matplotlib.pyplot as plt
import numpy as np

def create_accuracy_chart(accuracies, filename='model_accuracy_comparison_ascending.png'):
    # Sort accuracies in ascending order
    sorted_accuracies = dict(sorted(accuracies.items(), key=lambda item: item[1]))
    
    # Prepare data for plotting
    models = list(sorted_accuracies.keys())
    accuracy_values = list(sorted_accuracies.values())

    # Create bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, accuracy_values)

    # Customize the plot
    plt.title('Model Accuracy Comparison (Ascending Order)', fontsize=16)
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1)  # Set y-axis limit from 0 to 1 for accuracy

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Adjust layout
    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_precision_recall_f1_chart(results, filename='precision_recall_f1_comparison.png'):
    models = list(results.keys())
    
    # Function to safely get metric, defaulting to 0 if not found
    def get_metric(model, metric):
        report = results[model]['classification_report']
        return report.get('1.0', {}).get(metric, 0)

    precision = [get_metric(model, 'precision') for model in models]
    recall = [get_metric(model, 'recall') for model in models]
    f1_score = [get_metric(model, 'f1-score') for model in models]

    x = np.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    rects2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    rects3 = ax.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Precision, Recall, and F1-Score by Model (Class 1.0)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()

    # Add value labels on top of each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_hyperparameter_sensitivity_chart(results, model_name, param_name, filename=None):
    cv_results = results[model_name]['cv_results']
    
    param_values = []
    mean_scores = []
    for result in cv_results:
        if param_name in result['params']:
            param_values.append(result['params'][param_name])
            mean_scores.append(result['mean_cv_score'])
    
    def sort_key(item):
        return (item[0] is None, item[0])

    sorted_data = sorted(zip(param_values, mean_scores), key=sort_key)
    param_values, mean_scores = zip(*sorted_data)
    
    plt.figure(figsize=(12, 7))  # Increased figure size
    plt.plot(range(len(param_values)), mean_scores, marker='o')
    plt.title(f'{model_name}: Performance vs {param_name}', fontsize=16)
    plt.xlabel(param_name, fontsize=14)
    plt.ylabel('Mean CV Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.xticks(range(len(param_values)), [str(val) for val in param_values], rotation=45, ha='right')
    
    # Adjust y-axis to make room for labels
    plt.ylim(min(mean_scores) * 0.95, max(mean_scores) * 1.05)

    for i, (x, y) in enumerate(zip(param_values, mean_scores)):
        plt.annotate(f'{y:.4f}', (i, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    plt.tight_layout(pad=3.0)  # Added padding
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Hyperparameter sensitivity chart for {model_name} saved as '{filename}'")
    else:
        plt.show()
    
    plt.close()


result_files = [
    'DecisionTree_results.json',
    'KNN_results.json',
    'LogisticRegression_results.json',
    'NaiveBayes_results.json',
    'RandomForest_results.json',
    'SVM_results.json',
    'XGBoost_results.json'
]

results = {}

for file in result_files:
    with open(file, 'r') as f:
        data = json.load(f)
        model_name = file.split('_')[0]
        results[model_name] = data

# accuracies = {model: data['test_accuracy'] for model, data in results.items()}
# create_accuracy_chart(accuracies)

# create_precision_recall_f1_chart(results)

create_hyperparameter_sensitivity_chart(results, 'LogisticRegression', 'C', 'logistic_regression_C_sensitivity.png')
create_hyperparameter_sensitivity_chart(results, 'RandomForest', 'n_estimators', 'random_forest_n_estimators_sensitivity.png')
create_hyperparameter_sensitivity_chart(results, 'SVM', 'C', 'svm_C_sensitivity.png')
create_hyperparameter_sensitivity_chart(results, 'KNN', 'n_neighbors', 'knn_n_neighbors_sensitivity.png')
create_hyperparameter_sensitivity_chart(results, 'DecisionTree', 'max_depth', 'decision_tree_max_depth_sensitivity.png')
create_hyperparameter_sensitivity_chart(results, 'XGBoost', 'max_depth', 'xgboost_max_depth_sensitivity.png')
import json
import matplotlib.pyplot as plt
import numpy as np

# Load the results
with open('results/balanced_dataset_results.json', 'r') as f:
    balanced_results = json.load(f)

with open('results/unbalanced_dataset_results.json', 'r') as f:
    unbalanced_results = json.load(f)

# List of models
models = list(balanced_results.keys())

# Metrics to compare
metrics = ['accuracy', 'precision', 'recall', 'f1-score']

def get_metric(results, model, metric):
    if metric == 'accuracy':
        return results[model]['accuracy']
    else:
        return results[model]['classification_report']['1.0'][metric]

def create_comparison_chart(metric):
    # Get scores and sort models based on balanced dataset performance
    balanced_scores = [get_metric(balanced_results, model, metric) for model in models]
    unbalanced_scores = [get_metric(unbalanced_results, model, metric) for model in models]
    
    # Sort models and scores based on balanced dataset performance
    sorted_data = sorted(zip(models, balanced_scores, unbalanced_scores), key=lambda x: x[1])
    sorted_models, sorted_balanced, sorted_unbalanced = zip(*sorted_data)

    x = np.arange(len(sorted_models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))  # Increased figure size
    rects1 = ax.bar(x - width/2, sorted_balanced, width, label='Balanced', alpha=0.8)
    rects2 = ax.bar(x + width/2, sorted_unbalanced, width, label='Unbalanced', alpha=0.8)

    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} Comparison: Balanced vs Unbalanced (Class 1)')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, rotation=45, ha='right')
    ax.legend()

    # Adjust y-axis to make room for labels
    ax.set_ylim(0, max(max(sorted_balanced), max(sorted_unbalanced)) * 1.2)

    # Function to add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=0)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout(pad=3.0)  # Added padding

    plt.savefig(f'{metric}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create comparison charts for each metric
for metric in metrics:
    create_comparison_chart(metric)

print("Comparison charts have been saved as PNG files.")

# Create a summary table
def get_sorted_data(metric):
    balanced_scores = [get_metric(balanced_results, model, metric) for model in models]
    unbalanced_scores = [get_metric(unbalanced_results, model, metric) for model in models]
    return [x for _, x in sorted(zip(balanced_scores, models))], sorted(balanced_scores), [x for _, x in sorted(zip(balanced_scores, unbalanced_scores))]

sorted_models, sorted_balanced_accuracy, sorted_unbalanced_accuracy = get_sorted_data('accuracy')

summary_data = [
    ['Model'] + sorted_models,
    ['Balanced Accuracy'] + sorted_balanced_accuracy,
    ['Unbalanced Accuracy'] + sorted_unbalanced_accuracy,
]

for metric in ['precision', 'recall', 'f1-score']:
    _, sorted_balanced, sorted_unbalanced = get_sorted_data(metric)
    summary_data.append([f'Balanced {metric.capitalize()}'] + sorted_balanced)
    summary_data.append([f'Unbalanced {metric.capitalize()}'] + sorted_unbalanced)

fig, ax = plt.subplots(figsize=(16, 8))
ax.axis('off')
table = ax.table(cellText=summary_data[1:],  # Data rows
                 colLabels=summary_data[0],  # Header row
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.5)

plt.savefig('summary_table.png', dpi=300, bbox_inches='tight')
plt.close()

print("Summary table has been saved as 'summary_table.png'.")
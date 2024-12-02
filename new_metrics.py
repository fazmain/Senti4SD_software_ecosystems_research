from sklearn.metrics import classification_report, precision_recall_fscore_support
import pandas as pd
import json

# Load the predictions
input_file = 'GitHub/gh_validation_predictions.csv'
predictions_data = pd.read_csv(input_file)

# Define ground truth and predictions
true_labels = predictions_data['label']  # 'sentiment' for SO and Jira, 'label' for GitHub
predicted_labels = predictions_data['predicted_sentiment']

# Generate classification report
report = classification_report(true_labels, predicted_labels, labels=["positive", "neutral", "negative"], output_dict=True)

# Compute micro-averages
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, predicted_labels, labels=["positive", "neutral", "negative"], average='micro'
)

# Add micro-averages to the report
report["micro avg"] = {
    "precision": precision,
    "recall": recall,
    "f1-score": f1,
    "support": len(true_labels)
}

# Print the classification report
print("Classification Report:")
print(report)

# Save the metrics as a JSON file
output_file = 'metrics_report_gh.json'
with open(output_file, 'w') as f:
    json.dump(report, f, indent=4)

print(f"Metrics saved as {output_file}.")

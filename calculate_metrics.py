from sklearn.metrics import classification_report
import pandas as pd

# Load the predictions

input_file = 'gh_validation_predictions_default.csv'
predictions_data = pd.read_csv(input_file)  

# Define ground truth and predictions
true_labels = predictions_data['sentiment'] ## 'sentiment' for the stackoverflow and jira dataset, 'label' for the github dataset
predicted_labels = predictions_data['predicted_sentiment']

# Print classification metrics
report = classification_report(true_labels, predicted_labels, labels=["positive", "neutral", "negative"], output_dict=True)
print("Classification Report:")
print(report)

output_file = 'metrics_report_gh_default.json'

# Save the metrics as a JSON file for future reference
import json
with open(output_file, 'w') as f:
    json.dump(report, f, indent=4)

print(f"Metrics saved as {output_file}.")

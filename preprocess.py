import pandas as pd

# Function to preprocess the data
def preprocess_data(data):
    # Map integer labels to sentiment categories
    label_mapping = {
        1: "positive",
        0: "neutral",
        -1: "negative"
    }
    # Apply the mapping to the label column
    data['sentiment'] = data['label'].map(label_mapping)
    # Drop the original label column to avoid confusion
    data = data.drop(columns=['label'])
    return data

# Paths to your input `.pkl` files
train_file_path = 'so-train.pkl'  
test_file_path = 'so-test.pkl'    

# Load the `.pkl` files
train = pd.read_pickle(train_file_path)
test = pd.read_pickle(test_file_path)

# Preprocess the datasets
preprocessed_jira_train = preprocess_data(train)
preprocessed_jira_test = preprocess_data(test)

output_test = 'preprocessed_so_test.csv'
output_train = 'preprocessed_so_train.csv'

# Save the preprocessed datasets to CSV
preprocessed_jira_train.to_csv(output_train, index=False)
preprocessed_jira_test.to_csv(output_test, index=False)

print("Preprocessing complete.")
print(f"Train data saved as {output_train}")
print(f"Test data saved as {output_test}")

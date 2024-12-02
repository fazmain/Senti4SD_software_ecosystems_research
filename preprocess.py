import pandas as pd

# Function to preprocess the data only if the labels are mapped to integers
# def preprocess_data(data):
#     # Map integer labels to sentiment categories
#     label_mapping = {
#         1: "positive",
#         0: "neutral",
#         -1: "negative"
#     }
#     # Apply the mapping to the label column
#     data['sentiment'] = data['label'].map(label_mapping)
#     # Drop the original label column to avoid confusion
#     data = data.drop(columns=['label'])
#     return data

# Paths to your input `.pkl` files
train_file_path = 'gh-train.pkl'  
test_file_path = 'gh-test.pkl'    

# Load the `.pkl` files
train = pd.read_pickle(train_file_path)
test = pd.read_pickle(test_file_path)

print(test.head())

# Preprocess the datasets
# preprocessed_train = preprocess_data(train)
# preprocessed_test = preprocess_data(test)

output_test = 'preprocessed_gh_test.csv'
output_train = 'preprocessed_gh_train.csv'

# Save the preprocessed datasets to CSV
# preprocessed_train.to_csv(output_train, index=False)
# preprocessed_test.to_csv(output_test, index=False)

train.to_csv(output_train, index=False)
test.to_csv(output_test, index=False)

print("Preprocessing complete.")
print(f"Train data saved as {output_train}")
print(f"Test data saved as {output_test}")

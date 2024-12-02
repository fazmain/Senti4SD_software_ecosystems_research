import pandas as pd
import json

# Load the preprocessed training data

input_file = 'preprocessed_gh_train.csv'
train_data = pd.read_csv(input_file)  # Adjust file path if needed

# Prepare the chat format
def format_for_chat_model(row):
    return {
        "messages": [
            {"role": "system", "content": "You are a sentiment classification assistant who specialises on software engineering communication."},
            {"role": "user", "content": f"Classify the sentiment of the following text as positive, neutral, or negative: {row['sentence']}"},
            {"role": "assistant", "content": row['sentiment']} # `label` for the github dataset, `sentiment` for the stackoverflow dataset
        ]
    }

# Apply the formatting to all rows
chat_formatted_data = train_data.apply(format_for_chat_model, axis=1).tolist()

# Save to JSONL format
output_file = 'formatted_gh_train_chat.jsonl'

with open(output_file, 'w') as f:
    for entry in chat_formatted_data:
        f.write(json.dumps(entry) + '\n')

print(f"Chat-formatted training data saved as {output_file}.")

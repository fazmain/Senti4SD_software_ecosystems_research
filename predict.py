from openai import OpenAI
import pandas as pd

### PREDICTS USING FINE TUNED MODEL ###

# Initialises the OpenAI client. Uncomment and use your API key
# client = OpenAI(api_key="your-api-key")

# Model ID's for the fine-tuned models
# Stackoverflow: "ft:gpt-4o-mini-2024-07-18:personal:seco-so-1:AZbBLZfd"
# Github: "ft:gpt-4o-mini-2024-07-18:personal:seco-gh-1:AZlRYjr7"
# Jira: "ft:gpt-4o-mini-2024-07-18:personal:seco-jira-1:AZWACc5z"

# fine-tuned model ID
fine_tuned_model_id = "your-fine-tuned-model-id"

input_file_path = 'preprocessed_gh_test.csv'

# loading validation set
validation_data = pd.read_csv(input_file_path)  

# Format the validation set for testing
def format_for_testing(row):
    return {
        "messages": [
            {"role": "system", "content": "You are a sentiment classification assistant."},
            {"role": "user", "content": f"Classify the sentiment of the following text as positive, neutral, or negative: {row['sentence']}"}
        ]
    }

# Apply formatting
validation_formatted = validation_data.apply(format_for_testing, axis=1).tolist()

# Function to make predictions
def get_predictions(data, model_id):
    predictions = []
    for entry in data:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=entry["messages"],
                max_tokens=10,
                temperature=0
            )
            print(response)
            # Extract the prediction from the assistant's response
            prediction = response.choices[0].message.content.strip().lower()
            predictions.append(prediction)
        except Exception as e:
            print(f"Error generating prediction: {e}")
            predictions.append("error")  # Handle errors gracefully
    return predictions

# Get predictions for the validation set
validation_data['predicted_sentiment'] = get_predictions(validation_formatted, fine_tuned_model_id)

# Save predictions

output_file_path = 'gh_validation_predictions.csv'

validation_data.to_csv(output_file_path, index=False)
print(f"Predictions saved as {output_file_path}.")
from openai import OpenAI
import pandas as pd

### PREDICTS USING DEFAULT MODEL ###

# Initialises the OpenAI client. Uncomment and use your API key
# client = OpenAI(api_key="your_api_key")



# set the model to use
model = "gpt-4o-mini"

# Load the validation set

input_file_path = 'preprocessed_gh_test.csv'
validation_data = pd.read_csv(input_file_path)  

# formatting the testing data
def format_for_testing(row):
    return {
        "messages": [
            {"role": "system", "content": "You are a sentiment classification assistant who specialises in software engineering communication."},
            {"role": "user", "content": f"Classify the sentiment of the following text as positive, neutral, or negative: {row['sentence']}. Just type 'positive', 'neutral', or 'negative'."} # Added prompt for one word answer
        ]
    }

validation_formatted = validation_data.apply(format_for_testing, axis=1).tolist()

# making predictions
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
validation_data['predicted_sentiment'] = get_predictions(validation_formatted, model)

# Save predictions
output_file_path = 'gh_validation_predictions_default.csv'

validation_data.to_csv(output_file_path, index=False)
print(f"Predictions saved as {output_file_path}.")
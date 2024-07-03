import os
import openai
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Set up your OpenAI API key from the environment variable
api_key = os.getenv('API_KEY')
client = openai.OpenAI(api_key=api_key)

sample_size_train = 1000
sample_size_val = 200
sample_size_test = 200

# Load CONLL Dataset
data = load_dataset('conll2003')

# I optimized the system message by testing it on the browser with GPT 3.5
system_message = {
    "role": "system",
    "content": (
        "You are an expert in document classification. "
        "Classify the provided text into one of these categories by responding with only the category number: "
        "0 for World, 1 for Sport, 2 for Business, 3 for Technology, 4 for Other. "
        "Choose the most relevant category if the text fits into multiple categories. "
        "Do not provide any additional information or explanation in your response."
    )
}

# We make the API call with the system message, and user message is only the text to be labeled 
def get_document_label(text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[system_message, {"role": "user", "content": f"{text}"}],
            max_tokens=1,
            temperature=0.0
        )
        label = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        return label, tokens_used
    except Exception as e:
        print(f"Error in API call: {e}")
        return None, 0

# The main function of adding the label that is returned from the API to the dataset
def augment_dataset(dataset, type='train', sample_size=1000):
    dataset = dataset.shuffle(seed=23812).select(range(sample_size))
    augmented_data = []
    total_tokens = 0
    for sentence in tqdm(dataset, desc=f"Augmenting the {type} set"):
        sentence_txt = " ".join(sentence['tokens'])
        sentence_label, tokens_used = get_document_label(sentence_txt)
        total_tokens += tokens_used
        if sentence_label is not None:
            augmented_data.append({
                'tokens': sentence['tokens'],
                'ner_tags': sentence['ner_tags'],
                'sentence_label': sentence_label
            })
    print(f"Total tokens used for {type} set: {total_tokens}")
    return pd.DataFrame(augmented_data), total_tokens

# Augment datasets
train_augmented, train_tokens = augment_dataset(data['train'], 'train', sample_size=sample_size_train)
val_augmented, val_tokens = augment_dataset(data['validation'], 'validation', sample_size=sample_size_val)
test_augmented, test_tokens = augment_dataset(data['test'], 'test', sample_size=sample_size_test)

print(f"Total tokens used = {train_tokens + val_tokens + test_tokens}, approximate price: ${0.00050 * (train_tokens + val_tokens + test_tokens)/1000} assuming gpt-3.5-turbo-0125 on 02.07.2024")

# Save augmented datasets
train_augmented.to_csv('train_augmented.csv', index=False)
val_augmented.to_csv('val_augmented.csv', index=False)
test_augmented.to_csv('test_augmented.csv', index=False)

print("Dataset augmentation complete. Files saved as CSV.")

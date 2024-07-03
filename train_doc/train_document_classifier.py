import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import torch
import ast

# Load the augmented datasets
train_data = pd.read_csv('../data/train_augmented.csv')
val_data = pd.read_csv('../data/val_augmented.csv')
test_data = pd.read_csv('../data/test_augmented.csv')

# Function to preprocess the data
def preprocess_data(df):
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    df['text'] = df['tokens'].apply(lambda x: ' '.join(x))
    df['labels'] = df['sentence_label'].astype(int)
    return df[['text', 'labels']]

# Preprocess the data
train_data = preprocess_data(train_data)
val_data = preprocess_data(val_data)
test_data = preprocess_data(test_data)

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Set the format of the datasets
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Define metrics computation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results_document_classifier",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs_document_classifier',
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate on the test set
test_results = trainer.evaluate(tokenized_test)

print("Test results:", test_results)

# Save the model
trainer.save_model("./document_classifier_model")
print("Model saved to ./document_classifier_model")

# Printed output:
# Test results: {'eval_loss': 0.6955267190933228, 'eval_accuracy': 0.755, 'eval_f1': 0.7386042154566745, 'eval_runtime': 52.0812, 'eval_samples_per_second': 3.84, 'eval_steps_per_second': 0.25, 'epoch': 3.0}

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
from seqeval.metrics import classification_report, f1_score, accuracy_score,precision_score, recall_score
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from train import tokenize_and_align_labels

# Function to get predictions from a model
def get_predictions(model, dataset):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in dataset:
            inputs = {k: v.unsqueeze(0) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].unsqueeze(0)
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=2)
            predictions.extend(pred[0].tolist())
            true_labels.extend(labels[0].tolist())
    return predictions, true_labels


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load all models
model_sizes = [10, 30, 100, 300, 1000]
models = {}
for size in model_sizes:
    model_path = f"./models/models_datasize_{size}"
    models[size] = AutoModelForTokenClassification.from_pretrained(model_path)

# Load the test dataset
data = load_dataset('conll2003')
test_dataset = data["test"].shuffle(seed=23812).select(range(200)) # Change the size for different measures


# Tokenize the test dataset
tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=test_dataset.column_names)
tokenized_test_dataset.set_format("torch")

# Get predictions for all models
all_predictions = {}
for size, model in models.items():
    print(f"Getting predictions for model trained on {size} examples...")
    predictions, true_labels = get_predictions(model, tokenized_test_dataset)
    all_predictions[size] = (predictions, true_labels)

# Get the number of labels
num_labels = len(test_dataset.features['ner_tags'].feature.names)

# Convert numeric labels back to string labels
id2label = test_dataset.features['ner_tags'].feature.int2str

def convert_to_labels(predictions, true_labels):
    predictions_str = [id2label(p) if p != -100 and p < num_labels else "O" for p in predictions]
    true_labels_str = [id2label(t) if t != -100 and t < num_labels else "O" for t in true_labels]
    
    # Ensure predictions_str and true_labels_str have the same length
    min_len = min(len(predictions_str), len(true_labels_str))
    return predictions_str[:min_len], true_labels_str[:min_len]

# Calculate metrics
results = {}
for size, (predictions, true_labels) in all_predictions.items():
    pred_labels, true_labels_str = convert_to_labels(predictions, true_labels)
    
    # Wrap the labels in a list as seqeval expects a list of sequences
    pred_labels = [pred_labels]
    true_labels_str = [true_labels_str]
    
    results[size] = {
        "f1": f1_score(true_labels_str, pred_labels),
        "precision": precision_score(true_labels_str, pred_labels),
        "recall": recall_score(true_labels_str, pred_labels)
    }

# Print results
for size, metrics in results.items():
    print(f"\nResults for model trained on {size} examples:")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

# Plot F1 scores
plt.figure(figsize=(10, 6))
plt.plot(model_sizes, [results[size]['f1'] for size in model_sizes], marker='o')

plt.xlabel('Training Dataset Size')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Training Dataset Size')
plt.xscale('log')
plt.grid(True)
plt.savefig('f1_score_vs_dataset_size.png')
plt.close()

print("\nF1 score plot saved as 'f1_score_vs_dataset_size.png'")
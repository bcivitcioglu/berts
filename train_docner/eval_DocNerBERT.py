import pandas as pd
import torch
from transformers import BertTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import ast
from torch import nn
from transformers import BertModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from safetensors.torch import load_file

class DocNerBERT(nn.Module):
    def __init__(self, num_doc_labels, num_token_labels):
        super().__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)

        self.doc_classifier = nn.Linear(self.bert.config.hidden_size, num_doc_labels)
        self.token_classifier = nn.Linear(self.bert.config.hidden_size, num_token_labels)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, doc_labels=None, token_labels=None):
        outputs = self.bert(input_ids, attention_mask = attention_mask)

        # Document classification
        doc_output = outputs.pooler_output
        doc_output = self.dropout(doc_output)
        doc_logits = self.doc_classifier(doc_output)
        
        # Token classification
        token_output = outputs.last_hidden_state
        token_output = self.dropout(token_output)
        token_logits = self.token_classifier(token_output)

        if doc_labels is not None and token_labels is not None:
            doc_loss = self.loss(doc_logits, doc_labels)
            token_loss = self.loss(token_logits.view(-1, token_logits.shape[-1]), token_labels.view(-1))
            loss = doc_loss + token_loss

        return {
            'loss': loss,
            'doc_logits': doc_logits,
            'token_logits': token_logits
        }

def prepare_data(df):
    texts = df['tokens'].apply(lambda x: ' '.join(ast.literal_eval(x))).tolist()
    doc_labels = df['sentence_label'].tolist()
    token_labels = df['ner_tags'].apply(ast.literal_eval).tolist()
    
    return {
        "text": texts,
        "doc_labels": doc_labels,
        "token_labels": token_labels
    }

def tokenize_and_align_labels(examples, tokenizer, max_length=128):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
    
    labels = []
    for i, label in enumerate(examples["token_labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx] if word_idx < len(label) else -100)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["token_labels"] = labels
    tokenized_inputs["doc_labels"] = examples["doc_labels"]
    return tokenized_inputs

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    doc_logits, token_logits = logits
    doc_labels, token_labels = labels

    doc_predictions = np.argmax(doc_logits, axis=-1)
    token_predictions = np.argmax(token_logits, axis=-1)

    # Document classification metrics
    doc_accuracy = accuracy_score(doc_labels, doc_predictions)
    doc_f1 = f1_score(doc_labels, doc_predictions, average='weighted')
    doc_precision = precision_score(doc_labels, doc_predictions, average='weighted')
    doc_recall = recall_score(doc_labels, doc_predictions, average='weighted')

    # Token classification metrics
    mask = token_labels != -100
    token_predictions = token_predictions[mask]
    token_labels = token_labels[mask]
    token_accuracy = accuracy_score(token_labels, token_predictions)
    token_f1 = f1_score(token_labels, token_predictions, average='weighted')
    token_precision = precision_score(token_labels, token_predictions, average='weighted')
    token_recall = recall_score(token_labels, token_predictions, average='weighted')

    return {
        "doc_accuracy": doc_accuracy,
        "doc_f1": doc_f1,
        "doc_precision": doc_precision,
        "doc_recall": doc_recall,
        "token_accuracy": token_accuracy,
        "token_f1": token_f1,
        "token_precision": token_precision,
        "token_recall": token_recall
    }

def main():
    # Load the test dataset
    test_df = pd.read_csv('../data/test_augmented.csv')
    
    # Prepare test data
    test_data = prepare_data(test_df)
    
    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Create Dataset
    test_dataset = Dataset.from_dict(test_data)
    
    # Tokenize and align labels
    tokenized_test = test_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names
    )

    # Load the trained model
    num_doc_labels = len(test_df['sentence_label'].unique())
    num_token_labels = max(max(ast.literal_eval(x)) for x in test_df['ner_tags']) + 1
    
    # Initialize model
    model = DocNerBERT(num_doc_labels, num_token_labels)
    
    # Load the state dict from the safetensors file
    state_dict = load_file("./docnerbert_best_model/model.safetensors")
    
    # Load the state dict into the model
    model.load_state_dict(state_dict)

    # Define evaluation arguments
    eval_args = TrainingArguments(
        output_dir="./eval_results",
        per_device_eval_batch_size=8,
    )

    # Initialize Trainer for evaluation
    trainer = Trainer(
        model=model,
        args=eval_args,
        compute_metrics=compute_metrics
    )

    # Run evaluation
    eval_results = trainer.evaluate(tokenized_test)

    # Print results
    print("Test results:", eval_results)

    # Save results to CSV
    pd.DataFrame([eval_results]).to_csv('test_metrics.csv', index=False)
    print("Metrics saved to test_metrics.csv")

if __name__ == "__main__":
    main()
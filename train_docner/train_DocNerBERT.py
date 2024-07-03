import pandas as pd
import torch
from transformers import BertTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import ast
from torch import nn
import ast
from transformers import BertModel
from sklearn.metrics import accuracy_score, f1_score

# A simple BERT model that separates into two heads for both token and doc classification.
# One dropout layer added 
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


def main():
    # Read CSV files
    train_df = pd.read_csv('../data/train_augmented.csv')
    val_df = pd.read_csv('../data/val_augmented.csv')  # Load your separate validation set
    
    # Prepare data
    train_data = prepare_data(train_df)
    val_data = prepare_data(val_df)
    
    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Create Datasets
    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)
    
    # Tokenize and align labels for both datasets
    tokenized_train = train_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )

    # Initialize model
    num_doc_labels = len(train_df['sentence_label'].unique())
    num_token_labels = max(max(ast.literal_eval(x)) for x in train_df['ner_tags']) + 1
    model = DocNerBERT(num_doc_labels, num_token_labels)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save at the end of each epoch
        load_best_model_at_end=True,  # Load the best model at the end of training
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val
                )

    # Train the model
    trainer.train()

    # Save the best model
    trainer.save_model("./docnerbert_best_model")

if __name__ == "__main__":
    main()

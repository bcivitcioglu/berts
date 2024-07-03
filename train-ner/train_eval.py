from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments
)
from seqeval.metrics import f1_score
import numpy as np
import torch

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx != previous_word_idx:
                label_ids.append(label[word_idx] if word_idx is not None else -100)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

data = load_dataset('conll2003')
sizes = [10, 30, 100, 300, 1000]  
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

for size in sizes:
    print(f"Training on datasize: {size}.")
    
    train_dataset = data["train"].shuffle(seed=23812).select(range(size))
    train_labels = data['train'].features['ner_tags'].feature.names
    eval_dataset = data["validation"].shuffle(seed=23812).select(range(max(1, int(size/5))))
    
    test_sets = [
        data["test"].shuffle(seed=23812).select(range(200)),
        data["test"].shuffle(seed=23812).select(range(max(1, int(size/10)))),
        data["test"].shuffle(seed=23812).select(range(30))
    ]
    
    model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(train_labels))
    
    tokenized_datasets = [dataset.map(tokenize_and_align_labels, batched=True, remove_columns=dataset.column_names) 
                          for dataset in [train_dataset, eval_dataset] + test_sets]
    
    for dataset in tokenized_datasets:
        dataset.set_format("torch")
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=f"./models/model_datasize_{size}",
        eval_strategy="no", # Set to no because on my laptop the process is extremely slow, even for 1000 tokens
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets[0],
        eval_dataset=tokenized_datasets[1],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model(f"./models/model_datasize_{size}")
    print(f"Model saved to ./models/model_datasize_{size}")

    for i, test_set in enumerate(tokenized_datasets[2:], 1):
        predictions, labels, _ = trainer.predict(test_set)
        preds = np.argmax(predictions, axis=2)
        true_labels = [[train_labels[l] for l in label if l != -100] for label in labels]
        true_preds = [[train_labels[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(preds, labels)]
        f1 = f1_score(true_labels, true_preds)
        print(f"F1 score for test set {i}: {f1}")

print("Training and evaluation complete for all sizes.")
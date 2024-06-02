# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np

# Load the data
test_df = pd.read_csv('test.csv')
test_labels_df = pd.read_csv('test_labels.csv')

# Merge the datasets
merged_df = pd.merge(test_df, test_labels_df, on='id')

# Select the first 1000 records and the 'toxic' column
merged_df = merged_df.head(10000)
merged_df = merged_df[['id', 'comment_text', 'toxic']]
merged_df['toxic'] = merged_df['toxic'].replace(-1, 0)  # Replace -1 with 0

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

# Dataset class
class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).float().unsqueeze(0)
        return item

    def __len__(self):
        return len(self.labels)

# Splitting the data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    merged_df['comment_text'].tolist(),
    merged_df['toxic'].values.tolist(),
    test_size=0.2
)

# Tokenizing the texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Creating the datasets
train_dataset = HateSpeechDataset(train_encodings, train_labels)
val_dataset = HateSpeechDataset(val_encodings, val_labels)

# Custom metrics function
def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.sigmoid(torch.tensor(logits)).numpy() > 0.5
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# Setting up the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Adding early stopping callback
from transformers import EarlyStoppingCallback
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

# Creating the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback]
)

# Training the model
trainer.train()

# Save the trained model
model.save_pretrained('./trained_model3')
tokenizer.save_pretrained('./trained_model3')

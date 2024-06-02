from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the data
test_df = pd.read_csv('test.csv')
test_labels_df = pd.read_csv('test_labels.csv')

# Merge the datasets
merged_df = pd.merge(test_df, test_labels_df, on='id')
label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
merged_df[label_columns] = merged_df[label_columns].replace(-1, 0)

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_columns))

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples, padding="max_length", truncation=True)

# Dataset class
class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).float()
        return item

    def __len__(self):
        return len(self.labels)

# Splitting the data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    merged_df['comment_text'].tolist(),
    merged_df[label_columns].values.tolist(),
    test_size=0.2
)

# Tokenizing the texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Creating the datasets
train_dataset = HateSpeechDataset(train_encodings, train_labels)
val_dataset = HateSpeechDataset(val_encodings, val_labels)

# Custom loss function
def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.sigmoid(torch.tensor(logits)).numpy() > 0.5
    return {"accuracy": (predictions == labels).mean()}

# Setting up the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Creating the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Training the model
trainer.train()
# Save the trained model
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')
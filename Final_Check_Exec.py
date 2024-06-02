# predict_hate_speech.py
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import sys
from transformers import BertTokenizer, BertForSequenceClassification

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('./trained_model')
model = RobertaForSequenceClassification.from_pretrained('./trained_model')

def predict_hate_speech(paragraph):
    sentences = paragraph.split('.')
    predictions = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).detach().numpy()
        predictions.append(probs)
    return predictions

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_hate_speech.py '<paragraph>'")
        sys.exit(1)
    
    paragraph = sys.argv[1]
    predictions = predict_hate_speech(paragraph)
    
    for i, probs in enumerate(predictions):
        print(f"Sentence {i+1}: {probs}")

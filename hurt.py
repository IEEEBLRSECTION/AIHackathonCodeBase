from transformers import pipeline
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification


# Load a pre-trained text classification model (e.g., fine-tuned on offensive language)
classifier = pipeline('text-classification', 
                      tokenizer=BertTokenizer.from_pretrained('./trained_model3'),
                      model=BertForSequenceClassification.from_pretrained('./trained_model3'))



#classifier = pipeline('text-classification', tokenizer = BertTokenizer.from_pretrained('./trained_model3'),model = BertForSequenceClassification.from_pretrained.from_pretrained('./trained_model3'))
#classifier = pipeline('text-classification', model='unitary/toxic-bert')
# Your text extract
#tokenizer = BertTokenizer.from_pretrained('./trained_model')
#model = BertForSequenceClassification.from_pretrained('./trained_model')
text = "But he was dark guy. really i meant. He was fat and Ugly fellow. Its worst of all.Hi Ramesh"

# Split text into sentences
sentences = text.split('. ')

# Analyze each sentence
results = classifier(sentences)

# Display results
for sentence, result in zip(sentences, results):
    label = result['label']
    score = result['score']
    if label == 'toxic' and score > 0.5:  # Adjust the label and threshold as necessary
        print(f"Microaggressive sentence detected: {sentence} (Score: {score})")
    else:
        print(f"Microaggressive sentence not-detected: {sentence} (Score: {score})")

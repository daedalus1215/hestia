from transformers import pipeline

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_dir = './bert-base-uncased-serntiment-model'
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tok   = AutoTokenizer.from_pretrained(model_dir)
classifier = pipeline("text-classification", model=model, tokenizer=tok)
print(classifier(['I had a great time', 'My dog brings me happiness']))
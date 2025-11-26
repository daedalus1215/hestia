import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import evaluate

# Configuration - must match main.py
label2id = {"Real": 0, "Fake": 1}
id2label = {0: "Real", 1: "Fake"}
model_ckpt = "distilbert-base-uncased"
training_dir = "train_dir"
batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model
# The model is saved in a checkpoint directory, not directly in train_dir
import os
checkpoint_path = os.path.join(training_dir, "checkpoint-400")
if not os.path.exists(checkpoint_path):
    # Try to find the latest checkpoint
    import glob
    checkpoints = glob.glob(os.path.join(training_dir, "checkpoint-*"))
    if checkpoints:
        checkpoint_path = max(checkpoints, key=os.path.getctime)
        print(f"Using checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found in {training_dir}")

print("Loading trained model...")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# Load and prepare test data (same process as main.py)
print("Loading and preparing test data...")
df = pd.read_excel('../resources/fake_news.xlsx', dtype={'title': str, 'text': str})
df = df.dropna()

# Recreate the same train/test/validation split
train, test = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
test, validation = train_test_split(test, test_size=1/3, stratify=test['label'], random_state=42)

# Create dataset
dataset = DatasetDict({
    "train": Dataset.from_pandas(train, preserve_index=False),
    "test": Dataset.from_pandas(test, preserve_index=False),
    "validation": Dataset.from_pandas(validation, preserve_index=False)
})

# Tokenization function (same as main.py)
def distilbert_tokenize(batch):
    texts = [str(text) if text is not None and pd.notna(text) else "" for text in batch['text']]
    return tokenizer(texts, padding=False, truncation=True)

# Label encoding function (same as main.py)
def encode_labels(example):
    label = example['label']
    if isinstance(label, (int, float)) and label in id2label:
        example['label'] = int(label)
        return example
    
    label_str = str(label).strip()
    if label_str not in label2id:
        label_str_lower = label_str.lower()
        if label_str_lower == "real":
            label_str = "Real"
        elif label_str_lower == "fake":
            label_str = "Fake"
        else:
            raise ValueError(f"Unknown label: '{label}'. Expected 'Real' or 'Fake'")
    example['label'] = label2id[label_str]
    return example

# Tokenize and encode the test dataset
print("Tokenizing and encoding test dataset...")
encoded_test = dataset['test'].map(distilbert_tokenize, batched=True, batch_size=None)
encoded_test = encoded_test.map(encode_labels)

# Setup evaluation metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    """Compute comprehensive metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    metrics = {}
    metrics.update(accuracy_metric.compute(predictions=predictions, references=labels))
    metrics.update(f1_metric.compute(predictions=predictions, references=labels, average="weighted"))
    metrics.update(precision_metric.compute(predictions=predictions, references=labels, average="weighted"))
    metrics.update(recall_metric.compute(predictions=predictions, references=labels, average="weighted"))
    
    return metrics

# Create trainer for evaluation
eval_args = TrainingArguments(
    output_dir=training_dir,
    per_device_eval_batch_size=batch_size,
    disable_tqdm=False,
)

trainer = Trainer(
    model=model,
    args=eval_args,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)

# Evaluate on test set
print("\n" + "="*50)
print("Evaluating on TEST set...")
print("="*50)
test_results = trainer.evaluate(encoded_test)
print("\nTest Set Metrics:")
for key, value in test_results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Get predictions for detailed analysis
print("\n" + "="*50)
print("Generating predictions for detailed analysis...")
print("="*50)
predictions = trainer.predict(encoded_test)
predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

# Classification report
print("\nClassification Report:")
print(classification_report(
    true_labels, 
    predicted_labels, 
    target_names=[id2label[i] for i in sorted(id2label.keys())],
    digits=4
))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(true_labels, predicted_labels)
print(cm)
print(f"\nConfusion Matrix (with labels):")
print(f"                Predicted")
print(f"              Real    Fake")
print(f"Actual Real    {cm[0][0]:5d}  {cm[0][1]:5d}")
print(f"       Fake    {cm[1][0]:5d}  {cm[1][1]:5d}")

# Per-class metrics
print("\n" + "="*50)
print("Per-Class Metrics:")
print("="*50)
for i, label_name in sorted(id2label.items()):
    mask = true_labels == i
    if mask.sum() > 0:
        accuracy = (predicted_labels[mask] == i).mean()
        print(f"{label_name}: {accuracy:.4f} accuracy ({mask.sum()} samples)")

print("\n" + "="*50)
print("Evaluation complete!")
print("="*50)


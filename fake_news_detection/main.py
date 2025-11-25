import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('../resources/fake_news.xlsx', dtype={'title': str, 'text': str})
df.isnull().sum()
df = df.dropna()

df.isnull().sum()

# label_counts = df['label'].value_counts(ascending=True)
# label_counts.plot.barh()
# plt.savefig("../resources/output.png")
# print(df)

# Convert to string and handle NaN values
df['title_tokens'] = df['title'].apply(lambda x: len(str(x).split()) * 1.5 if pd.notna(x) else 0)
df['text_tokens'] = df['text'].apply(lambda x: len(str(x).split()) * 1.5 if pd.notna(x) else 0)
print (df['title_tokens'].describe())
print (df['text_tokens'].describe())

print('---------------df-----------------')
print(df)

fig, ax = plt.subplots(1,2, figsize=(15,5))


ax[0].hist(df['title_tokens'], bins=50, color='skyblue')
ax[0].set_title('Title Tokens')

ax[1].hist(df['text_tokens'], bins=50, color='orange')
ax[1].set_title('Text Tokens')

plt.savefig("../resources/output.png")




# Data Preparation (Training, Testing, Validation)
from sklearn.model_selection import train_test_split
# 70% for training, 20% test, 10% validation
train, test = train_test_split(df, test_size=0.3, stratify=df['label'])
test, validation = train_test_split(test, test_size=1/3, stratify=test['label'])
print(train.shape, test.shape, validation.shape, df.shape)

# Need to get it ready for huggingface
from datasets import Dataset, DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_pandas(train, preserve_index=False),
    "test": Dataset.from_pandas(test, preserve_index=False),
    "validation": Dataset.from_pandas(validation, preserve_index=False)
})

print(dataset)






# Data Tokenization
from transformers  import AutoTokenizer
text = "Machine learning is awesome!! Thanks KGP Talkie."

model_ckpt = "distilbert-base-uncased"
distilbert_tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
distilbert_tokens = distilbert_tokenizer.tokenize(text)


model_ckpt = "google/mobilebert-uncased"
mobilebert_tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
mobilebert_tokens = mobilebert_tokenizer.tokenize(text)

model_ckpt = "huawei-noah/TinyBERT_General_4L_312D"
tinybert_tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
tinybert_tokens = tinybert_tokenizer.tokenize(text)

print('---------------distilbert_tokens-----------------')
print(distilbert_tokens)
print('---------------mobilebert_tokens-----------------')
print(mobilebert_tokens)
print('---------------tinybert_tokens-----------------')
print(tinybert_tokens)


def distilbert_tokenize(batch):
    # Handle None/NaN values by converting to empty string
    texts = [str(text) if text is not None and pd.notna(text) else "" for text in batch['text']]
    temp = distilbert_tokenizer(texts, padding=False, truncation=True)
    return temp

print('---------------distilbert_tokenize-----------------')
print(distilbert_tokenize(dataset['train'][:2]))

def mobilebert_tokenize(batch):
    temp = mobilebert_tokenizer(batch['text'], padding=True, truncation=True)
    return temp

print('---------------mobilebert_tokenize-----------------')
print(mobilebert_tokenize(dataset['train'][:2]))

def tinybert_tokenize(batch):
    temp = tinybert_tokenizer(batch['text'], padding=True, truncation=True)
    return temp

print('---------------tinybert_tokenize-----------------')
print(tinybert_tokenize(dataset['train'][:2]))

### Apply the map function
# (Tokenization will be applied after label2id is defined in Model Building section)







# Model Building

from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification
import torch

label2id = {"Real": 0, "Fake": 1}
id2label = {0: "Real", 1: "Fake"}
num_labels = len(label2id)

# Tokenize the dataset and encode labels
def encode_labels(example):
    label = example['label']
    # If already numeric, use it directly
    if isinstance(label, (int, float)) and label in id2label:
        example['label'] = int(label)
        return example
    
    # Convert label to string, strip whitespace, and map to ID
    label_str = str(label).strip()
    if label_str not in label2id:
        # Try with different case variations
        label_str_lower = label_str.lower()
        if label_str_lower == "real":
            label_str = "Real"
        elif label_str_lower == "fake":
            label_str = "Fake"
        else:
            raise ValueError(f"Unknown label: '{label}'. Expected 'Real' or 'Fake', got: {type(label)}")
    example['label'] = label2id[label_str]
    return example

encoded_dataset = dataset.map(distilbert_tokenize, batched=True, batch_size=None)
encoded_dataset = encoded_dataset.map(encode_labels)

model_ckpt = "distilbert-base-uncased"
mobilebert_model_ckpt = "google/mobilebert-uncased"
tinybert_model_ckpt = "huawei-noah/TinyBERT_General_4L_312D"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

distilbert_config = AutoConfig.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label)
distilbert_model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=distilbert_config).to(device)

mobilebert_config = AutoConfig.from_pretrained(mobilebert_model_ckpt, label2id=label2id, id2label=id2label)
mobilebert_model = AutoModelForSequenceClassification.from_pretrained(mobilebert_model_ckpt, config=mobilebert_config).to(device)

tinybert_config = AutoConfig.from_pretrained(tinybert_model_ckpt, label2id=label2id, id2label=id2label)
tinybert_model = AutoModelForSequenceClassification.from_pretrained(tinybert_model_ckpt, config=tinybert_config).to(device)

print('---------------distilbert_model-----------------')
print(distilbert_model.config)
print('---------------mobilebert_model-----------------')
print(mobilebert_model.config)
print('---------------tinybert_model-----------------')
print(tinybert_model.config)



# Model Training


import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

def compute_metrics_evaluate(eval_pred):
    f""" Compute metrics using evaluate library. 
    eval_pred is a tuple of predictions and labels, pass automatically by the trainer. 
    Predictions are the raw logits (model outputs before softmax). 
    Labels are the ground-truth numeric class IDs.
    np.argmax(predictions, axis=1) converts the logits to predicted class IDs by picking the index of the max logit (the most likely class).
    accuracy.compute() calls the built-in hugging face metric to compute accuracy. returns a dict like {"accuracy": 0.9}
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


from transformers import Trainer
from transformers import TrainingArguments
batch_size = 64
training_dir = "train_dir"
training_args = TrainingArguments(
    output_dir=training_dir,
    overwrite_output_dir=True,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    disable_tqdm=False,
)

print('---------------training_args-----------------')
print(training_args)

trainer = Trainer(
    model=distilbert_model,
    args=training_args,
    compute_metrics=compute_metrics_evaluate,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation'],
    processing_class=distilbert_tokenizer,
)

trainer.train()
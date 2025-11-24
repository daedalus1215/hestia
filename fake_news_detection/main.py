import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('../resources/fake_news.xlsx', dtype={'title': str, 'text': str})

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


### Let's start to Tokenize the data
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


### Time to tokenize

def distilbert_tokenize(batch):
    temp = distilbert_tokenizer(batch['text'], padding=True, truncation=True)
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
# encoded_dataset = dataset.map(tokeni)




import transformers import AutoModel
import torch

label2id = {"Real": 0, "Fake": 1}
id2label = {0: "Real", 1: "Fake"}
model_ckpt = "distilbert-base-uncased"

num_labels = len(label2id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = AutoConfig.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config).to(device)


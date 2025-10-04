import pandas as pd

df = pd.read_csv('twitter_multi_class_sentiment.csv')
print('---------------head-----------------')
print(df.head())
print('---------------columns-----------------')
print(df.columns)
print('---------------shape-----------------')
print(df.shape)
print('---------------info-----------------')
print(df.info())
print('---------------df-----------------')
print(df)
print('---------------describe-----------------')
print(df.describe())
print('---------------isnull().sum()-----------------')
print(df.isnull().sum())
print('---------------label value counts-----------------')
print(df['label'].value_counts(ascending=True))
df['Words per Tweet'] = df['text'].str.split().apply(len)
print('---------------Tokenization-----------------')


# from transformers import AutoTokenizer
# model_ckpt = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# print(tokenizer.tokenize("Hello, how are you?"))
# print(tokenizer.convert_tokens_to_ids("Hello, how are you?"))
# print(tokenizer.convert_ids_to_tokens(100))
# print(tokenizer.decode(100))

#  Play around with parsing data into a dataset dictionary
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.3, stratify=df['label'])
test, validation = train_test_split(test, test_size=1/3, stratify=test['label'])
print('train.shape', train.shape)
print('test.shape', test.shape)
print('validation.shape', validation.shape)


from datasets import Dataset, DatasetDict

dataset = DatasetDict({
    'train': Dataset.from_pandas(train, preserve_index=False),
    'test': Dataset.from_pandas(test, preserve_index=False),
    'validation': Dataset.from_pandas(validation, preserve_index=False)
})

# print(dataset)

# Tokenization of the data
from transformers import AutoTokenizer
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    temp = tokenizer(batch['text'], padding=True, truncation=True)
    return temp

encoded_dataset = dataset.map(tokenize, batched=True, batch_size=None)
print(encoded_dataset)
print('---------------train check-----------------')

# print(tokenize(dataset['train'][:2])) 
# print(dataset['train'][0])
emotion_encoded = dataset.map(tokenize, batched=True, batch_size=None)
print('---------------emotion_encoded-----------------')
print(emotion_encoded)

label2id = { x['label_name']:x['label'] for x in dataset['train']}
id2label = {v: k for k, v in label2id.items()}

print('---------------label2id-----------------')
print(label2id)
print('---------------id2label-----------------')
print(id2label)
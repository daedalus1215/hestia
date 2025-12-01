import pandas as pd
import json
import requests


train = pd.read_csv("data/train.bio", sep="\t", header=None)
train_tokens = train[1].tolist() 
train_tags = train[0].tolist()

# print(train.head())
# print(f"\ntrain_tokens: {train_tokens[:10]}")
# print(f"Tags: {tags[:10]}")

# print("---------------len(train_tokens)-----------------")
# print(len(train_tokens))
# print("---------------len(tags)-----------------")
# print(len(tags))

test = pd.read_csv("data/test.bio", sep="\t", header=None)
test_tokens = test[1].tolist()
test_tags = test[0].tolist()

print("---------------len(test_tokens)-----------------")
print(len(test_tokens))
print("---------------len(test_tags)-----------------")
print(len(test_tags))



from datasets import Dataset, DatasetDict

train_df = pd.DataFrame({'tokens': train_tokens, 'ner_tags_str': train_tags})
train_dataset = Dataset.from_pandas(train_df)

test_df = pd.DataFrame({'tokens': test_tokens, 'ner_tags_str': test_tags})
test_dataset = Dataset.from_pandas(test_df)

# Being a little lazy and using the test dataset as the validation dataset. I could have used a separate validation dataset. 
dataset = DatasetDict({'train': train_dataset, 'test': test_dataset, 'validation': test_dataset})

print("---------------dataset-----------------")
print(dataset)
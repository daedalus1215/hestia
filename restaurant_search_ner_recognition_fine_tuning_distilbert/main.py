import pandas as pd
import json
import requests


train = pd.read_csv("https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/mit_restaurant_search_ner/train.bio", sep="\t", header=None)
print(train.head())

# Separate tokens and tags into two arrays
tokens = train[1].tolist()  # Column 1 contains tokens
tags = train[0].tolist()    # Column 0 contains tags

print(f"\nTokens: {tokens[:10]}")
print(f"Tags: {tags[:10]}")

print("---------------len(tokens)-----------------")
print(len(tokens))
print("---------------len(tags)-----------------")
print(len(tags))



from datasets import Dataset, DatasetDict

df = pd.DataFrame({'tokens': tokens, 'ner_tags_str': tags})
dataset = Dataset.from_pandas(df)

dataset = DatasetDict({'train': dataset})

print("---------------dataset-----------------")
print(dataset)


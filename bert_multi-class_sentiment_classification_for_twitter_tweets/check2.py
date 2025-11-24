from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import pandas as pd

df = pd.read_csv('twitter_multi_class_sentiment.csv')
# 1) Split into train/val; keep test separate if you have it
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# 2) Convert DataFrames -> Hugging Face Datasets
ds = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
    # "test": Dataset.from_pandas(test_df.reset_index(drop=True))  # if you have one
})

# 3) Tokenize into model-ready format (input_ids, attention_mask, labels)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    toks = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    toks["labels"] = batch["label"]
    return toks

ds = ds.map(tokenize, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in ("text","label")])
ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
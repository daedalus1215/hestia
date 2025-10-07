from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("distilbert-base-uncased")
print('---------------model-----------------')
print(model)
print('---------------model.config-----------------')
print(model.config)

print('---------------model.config.hidden_size-----------------')
print(model.config.hidden_size)
print('---------------model.config.num_labels-----------------')
print(model.config.num_labels)
print('---------------model.config.num_labels-----------------')
# import pandas as pd

# df = pd.read_csv('twitter_multi_class_sentiment.csv')

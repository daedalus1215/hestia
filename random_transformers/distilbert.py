from transformers import pipeline
import pandas as pd

nlp = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f",   # or omit and let it use latest
    device_map="auto"     # will pick GPU if available
)

result =  nlp("this is great")
print(result)
pd.DataFrame(result)
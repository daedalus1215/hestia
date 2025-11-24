from transformers import pipeline
import pandas as pd

text = """
Dear Amazon, last week I ordered an Optimus Price 
action figure from your online store. Unfortunately when
I opened the package, I discovered that I had been sent
an action figure of Megatron instead!
"""

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

outputs = summarizer(text)

pd.DataFrame([outputs])

print(outputs)
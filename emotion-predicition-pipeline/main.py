from transformers import pipeline

nlp = pipeline(
    # "sentiment-analysis",
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f",   # or omit and let it use latest
    device_map="auto"     # will pick GPU if available
)
print(nlp("this is great"))

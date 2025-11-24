from transformers import pipeline
import pandas as pd

nlp = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions"
)

result =  nlp("Sally, I think I am panicking. I can't find my dog!")

print(result)

pd.DataFrame(result)
from transformers import pipeline
import pandas as pd

text = """
Dear Amazon, last week I ordered an Optimus Price 
action figure from your online store. Unfortunately when
I opened the package, I discovered that I had been sent
an action figure of Megatron instead!
"""

translate_eng_to_fr = pipeline("translation_en_to_fr")

outputs = translate_eng_to_fr(text)

pd.DataFrame([outputs])

print(outputs)

translate_fr_to_en = pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en")
outputs_fr_to_en = translate_fr_to_en(outputs[0]['translation_text'])
print(outputs_fr_to_en)

from transformers import set_seed, pipeline
set_seed(0)


generator = pipeline("text-generation", model="gpt2-large")

prompt = "A wild lion entered a store"

outputs = generator(prompt, max_length=500, num_return_sequences=1)

print(outputs)
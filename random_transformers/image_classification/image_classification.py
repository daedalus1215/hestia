from PIL import Image
import requests
from transformers import pipeline

url = "https://imgs.search.brave.com/fM5fe860LYWvURuki4Vj3MAu6_2Tpv67akybmFBpwYU/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly93d3cu/YXNwY2Eub3JnL3Np/dGVzL2RlZmF1bHQv/ZmlsZXMvY2F0LWNh/cmVfYWdncmVzc2lv/bi1pbi1jYXRzX21h/aW4taW1hZ2UuanBn"
image = Image.open(requests.get(url, stream=True).raw)
classifier = pipeline("image-classification")
outputs = classifier(image)
print(outputs)
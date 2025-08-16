from PIL import Image
import requests
from transformers import pipeline

url = "https://imgs.search.brave.com/ko5aPhJOuH8q5NdgKKjyeYGZ2OazPgJxl00EG_LIdco/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9zdGF0/aWMudmVjdGVlenku/Y29tL3N5c3RlbS9y/ZXNvdXJjZXMvdGh1/bWJuYWlscy8wNTYv/NjQ0LzA0OS9zbWFs/bC9hLW1hbi13aXRo/LWEtZnVsbC13ZWxs/LWdyb29tZWQtYmVh/cmQtYW5kLWludGVu/c2UtZXllcy1wb3Nl/cy1pbi10cmFkaXRp/b25hbC1hdHRpcmUt/aGlzLXNlcmlvdXMt/ZXhwcmVzc2lvbi1z/dWdnZXN0cy1hLWhp/c3RvcmljYWwtY29u/dGV4dC1hbWlkc3Qt/YS1ydXN0aWMtd29v/ZGVuLWJhY2tkcm9w/LXBob3RvLmpwZw"
image = Image.open(requests.get(url, stream=True).raw)

segmenter = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")

outputs = segmenter(image)

print(outputs)
outputs[1]['mask']
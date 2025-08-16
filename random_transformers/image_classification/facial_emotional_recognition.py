from PIL import Image
import requests
from transformers import pipeline

url = "https://imgs.search.brave.com/RLOqlKgjCPXo8hBngVheVIm_vP9zqwl8dMk-oj-KXuM/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9tZWRp/YS5nZXR0eWltYWdl/cy5jb20vaWQvMTQx/MzM2ODU2NS9waG90/by9oYXBweS10cmVu/ZHktYW5kLWZ1bmt5/LWJsYWNrLW1hbi1s/b29raW5nLWNoZWVy/ZnVsLXdpdGgtYS1i/aWctc21pbGUtb24t/aGlzLWZhY2Utc3Rh/bmRpbmcuanBnP3M9/NjEyeDYxMiZ3PTAm/az0yMCZjPXpCZ3pn/NnJBMVNzQ1REaWQ5/WWExZU53OHhOTWRD/Sk4zWDVqRG4tVDZj/bFk9"
image = Image.open(requests.get(url, stream=True).raw)
classifier = pipeline("image-classification", model="Rajaram1996/FacialEmoRecog")
outputs = classifier(image)
print(outputs)
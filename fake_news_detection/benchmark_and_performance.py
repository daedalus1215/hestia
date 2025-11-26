from transformers import pipeline
classifier = pipeline("text-classification", model="fake_news")

print(classifier("On January 1st, 2025, chickens crossed the road and caused a traffic jam."))
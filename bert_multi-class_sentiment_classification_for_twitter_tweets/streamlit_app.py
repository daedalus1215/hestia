import streamlit as st
import pandas as pd
import numpy as np

from transformers import pipeline

st.title("Fine Tuning BERT for Twitter Tweets for multi class sentiment classification")

classifier = pipeline("text-classification", model='bert-base-uncased-serntiment-model')


text = st.text_area("Enter your tweet here")

if st.button('Predict'):
    result = classifier(text)
    st.write(result)
# Default Libraries
import streamlit as st
import pandas as pd
import numpy as np

# Model Libraries
import pickle
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import string
import re


def text_cleaning(text):
    text = str(text) # Convert into string
    text = text.lower() # Lowercasing text
    text = re.sub(r'\d+', '', text) # Remove numbers
    tokenizer = RegexpTokenizer(r'\w+') # tokenize text & remove punctuations
    tokened_text = tokenizer.tokenize(text)
    
    text = [w for w in tokened_text if w not in stopwords.words('english')] # Remove stop words
    lemmatizer = WordNetLemmatizer()
    new_text = [lemmatizer.lemmatize(w) for w in text]
    
    return ' '.join(new_text)


with open('lr-sgd-us-pipeline.pickle','rb') as modelFile:
    model = pickle.load(modelFile)


def classify_message(model, message):
    label = model.predict([message])[0]
    return label


def run():
    st.title("Hotel Review Rating Prediction")
    # message_text = st.text_input("Enter a review")
    message_text = st.text_area("Enter a review", height=300)

    
    if st.button("Predict"):
            output = classify_message(model, message_text)
            st.write('This is a ' + str(output) + ' star hotel review!')


if __name__ == '__main__':
    run()

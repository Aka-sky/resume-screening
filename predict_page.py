import streamlit as st
import pickle
import numpy as np
import pdfplumber
import re
from io import StringIO

def load_model():
    with open('predictor.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

saved_model = data["model"]
saved_le = data["le"]
saved_wv = data["word_vectorizer"]

# Text cleaning helper
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

# extract text from pdf
def extract_text(feed):
    text = ''
    with pdfplumber.open(feed) as pdf:
        pages = pdf.pages
        for p in pages:
            text += p.extract_text()

    return text

uploaded_files = None

def show_predict_page():
    st.title("Resume Category Prediction Software")

    st.subheader("Upload Resume PDFs below")

    global uploaded_files
    uploaded_files = st.file_uploader('Resume pdf', type='pdf', accept_multiple_files=True)

    onClick = st.button("Predict")

    if onClick:
        # first extact text from PDF
        texts = []
        if uploaded_files is not None:
            for upload in uploaded_files:
                texts.append(extract_text(upload))

        # Clean Text
        texts = [cleanResume(text) for text in texts]

        # Apply model
        vectorized_text = saved_wv.transform(texts)
        pred = saved_model.predict(vectorized_text)
        predictions = saved_le.inverse_transform(pred)

        # st.subheader(f"The estimated category is {prediction}.")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("File")
            for upload in uploaded_files:
                st.text(upload.name)
        
        with col2:
            st.subheader("Predicted Category")
            for prediction in predictions:
                st.text(prediction)
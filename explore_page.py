import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from matplotlib.gridspec import GridSpec


def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

@st.cache
def load_data():
    resumeDataSet = pd.read_csv('ResumeDataset.csv' ,encoding='utf-8')
    resumeDataSet['cleaned_resume'] = ''

    resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))

    return resumeDataSet

def show_explore_page():
    st.title("Explore page")

    st.markdown(
        """#### Resume Dataset from [Kaggle](https://www.kaggle.com/dhainjeamita/updatedresumedataset/code)"""
    )

    resumeDataSet = load_data()

    st.subheader("Categories in Dataset")
    plt.figure(figsize=(15,15))
    plt.xticks(rotation=90)
    sns.countplot(y="Category", data=resumeDataSet)
    st.pyplot(plt)

    st.subheader("Distribution of Categories")
    targetCounts = resumeDataSet['Category'].value_counts()
    targetLabels  = resumeDataSet['Category'].unique()
    # Make square figures and axes
    plt.figure(1, figsize=(25,25))
    the_grid = GridSpec(2, 2)
    cmap = plt.get_cmap('coolwarm')
    colors = [cmap(i) for i in np.linspace(0, 1, 3)]
    plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')
    source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, colors=colors)
    st.pyplot(plt)
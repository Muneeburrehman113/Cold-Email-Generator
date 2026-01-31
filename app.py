import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("portfolio_data.csv")

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["description"])

def query_portfolio(job_description, top_k=3):
    query_vec = vectorizer.transform([job_description])
    scores = (tfidf_matrix @ query_vec.T).toarray().ravel()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return df.iloc[top_idx]

def generate_email(job_desc, matches):
    projects = "\n".join(
        f"- {row['title']} ({row['skills']})"
        for _, row in matches.iterrows()
    )

    return f"""
Subject: Application for Python Developer Role

Hi Hiring Team,

I am excited to apply for the Python Developer position.

Relevant projects:
{projects}

Best regards,
[Your Name]
"""

st.title("AI Cold Email Generator")

job_desc = st.text_area("Enter Job Description")

if st.button("Generate Email"):
    matches = query_portfolio(job_desc)
    st.subheader("Relevant Projects")
    st.dataframe(matches)
    st.subheader("Generated Email")
    st.code(generate_email(job_desc, matches))

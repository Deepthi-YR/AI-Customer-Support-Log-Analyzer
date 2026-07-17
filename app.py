# ==========================================================
# AI CUSTOMER SUPPORT LOG ANALYZER
# PGDBA CAPSTONE PROJECT
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------

st.set_page_config(
    page_title="AI Customer Support Log Analyzer",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#F8F9FA;
}

.metric-container{
    background:#ffffff;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 3px 8px rgba(0,0,0,0.08);
}

h1{
    color:#0E4C92;
}

h2{
    color:#0E4C92;
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("customer_support_cleaned.xls")

    return df

df = load_data()

# ----------------------------------------------------------
# LOAD ML MODEL
# ----------------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load("best_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------

st.sidebar.image(
    "https://img.icons8.com/color/96/customer-support.png",
    width=90
)

st.sidebar.title("AI Customer Support")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Home",

        "📊 Dashboard",

        "📈 Business Insights",

        "🤖 AI Ticket Predictor",

        "😊 Sentiment Analysis",

        "📂 Data Explorer",

        "ℹ️ About"

    ]

)

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.title("🎧 AI Customer Support Log Analyzer")

    st.markdown("---")

    col1, col2 = st.columns([2,1])

    with col1:

        st.markdown("""
### 📌 Project Objective

This project analyzes customer support tickets using Business Analytics
and Machine Learning.

The application enables businesses to:

- Monitor customer support KPIs
- Analyze ticket categories
- Track customer satisfaction
- Identify root causes
- Predict ticket categories using AI
- Generate business insights

---
### 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- Plotly
- Scikit-Learn
- TF-IDF
- XGBoost
- Machine Learning

---
### 🤖 Machine Learning

**Model Used**

✔ XGBoost Classifier

The model predicts customer support ticket categories
from customer ticket descriptions.

""")

    with col2:

        st.info("""
### Project Summary

Dataset Size

✔ 8,469 Tickets

Machine Learning

✔ XGBoost

Deployment

✔ Streamlit Cloud

Course

PGDBA Capstone Project

""")

    st.markdown("---")

    st.success("✅ AI Customer Support Log Analyzer successfully loaded.")

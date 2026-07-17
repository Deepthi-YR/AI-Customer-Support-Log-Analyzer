# ==========================================================
# AI CUSTOMER SUPPORT LOG ANALYZER
# Streamlit Application
# ==========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="AI Customer Support Log Analyzer",
    page_icon="🎧",
    layout="wide"
)

# ----------------------------------------------------------
# Load Data
# ----------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("customer_support_cleaned.csv")

df = load_data()

dashboard_df = pd.read_csv("dashboard_data.csv")
business_df = pd.read_csv("business_summary.csv")
product_df = pd.read_csv("product_summary.csv")

# ----------------------------------------------------------
# Load ML Model
# ----------------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("best_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------

st.sidebar.title("🎧 AI Customer Support")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dashboard",
        "📈 Business Insights",
        "🤖 AI Ticket Predictor",
        "📂 Data Explorer",
        "ℹ️ About"
    ]
)

# ==========================================================
# HOME
# ==========================================================

if page == "🏠 Home":

    st.title("🎧 AI Customer Support Log Analyzer")

    st.markdown("""
    ### Project Overview

    This application analyzes customer support tickets using
    Business Analytics and Machine Learning.

    ### Features

    - 📊 Business Dashboard
    - 📈 Customer Support Insights
    - 🤖 AI Ticket Category Prediction
    - 📂 Interactive Data Explorer
    - 📋 KPI Monitoring

    ### Machine Learning Model

    ✔ XGBoost Classifier

    ### Technologies Used

    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - XGBoost
    - TF-IDF
    """)

    st.success("Project Developed as part of PGDBA Capstone")

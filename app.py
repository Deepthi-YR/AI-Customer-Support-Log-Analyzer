# ==========================================================
# AI CUSTOMER SUPPORT LOG ANALYZER
# STREAMLIT APPLICATION
# PART 1A - HOME PAGE
# ==========================================================

import streamlit as st
import pandas as pd
import joblib
import os

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------

st.set_page_config(
    page_title="AI Customer Support Log Analyzer",
    page_icon="🎧",
    layout="wide"
)

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

@st.cache_data
def load_data():

    file_path = "customer_support_cleaned.csv"

    if not os.path.exists(file_path):
        st.error("❌ Dataset not found.")
        st.stop()

    df = pd.read_csv(file_path)

    return df


df = load_data()

# ----------------------------------------------------------
# LOAD MODEL FILES
# ----------------------------------------------------------

@st.cache_resource
def load_models():

    try:

        model = joblib.load("best_model.joblib")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        label_encoder = joblib.load("label_encoder.pkl")

        return model, vectorizer, label_encoder

    except:

        st.warning("Model files not found. AI Classifier will be unavailable.")

        return None, None, None


model, vectorizer, label_encoder = load_models()

# ----------------------------------------------------------
# DATA PREPARATION
# ----------------------------------------------------------

# Convert Date column safely

if "Date of Purchase" in df.columns:

    df["Date of Purchase"] = pd.to_datetime(
        df["Date of Purchase"],
        errors="coerce"
    )

# Numeric columns

numeric_columns = [
    "Customer Satisfaction Rating",
    "First Response Time",
    "Time to Resolution"
]

for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

# ----------------------------------------------------------
# KPI CALCULATIONS
# ----------------------------------------------------------

total_tickets = len(df)

closed_tickets = (
    df["Ticket Status"]
    .astype(str)
    .str.lower()
    .str.contains("closed")
    .sum()
)

open_tickets = (
    df["Ticket Status"]
    .astype(str)
    .str.lower()
    .str.contains("open")
    .sum()
)

avg_rating = round(
    df["Customer Satisfaction Rating"].mean(),
    2
)

# ----------------------------------------------------------
# TITLE
# ----------------------------------------------------------

st.title("🎧 AI Customer Support Log Analyzer")

st.caption(
    "Business Analytics Dashboard with AI-powered Ticket Classification"
)

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Page",

    [

        "Home",

        "Dashboard",

        "Business Insights",

        "AI Ticket Classifier",

        "Dataset Explorer",

        "About Project"

    ]

)

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "Home":

    st.header("🏠 Home")

    st.write("""

Welcome to the **AI Customer Support Log Analyzer**.

This application helps analyze customer support tickets,
generate business insights and automatically classify
customer complaints using Natural Language Processing (NLP)
and Machine Learning.

""")

    st.divider()

    # ------------------------------------------------------

    st.subheader("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Tickets",
        total_tickets
    )

    col2.metric(
        "Closed Tickets",
        closed_tickets
    )

    col3.metric(
        "Open Tickets",
        open_tickets
    )

    col4.metric(
        "Average Rating",
        avg_rating
    )

    st.divider()

    # ------------------------------------------------------

    st.subheader("Project Workflow")

    st.markdown("""

Customer Complaint

⬇️

Text Cleaning & Preprocessing

⬇️

TF-IDF Vectorization

⬇️

Random Forest Machine Learning Model

⬇️

Automatic Ticket Category Prediction

""")

    st.divider()

    # ------------------------------------------------------

    st.subheader("Technologies Used")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.write("🐍 Python")

        st.write("📊 Pandas")

    with c2:

        st.write("🤖 Scikit-learn")

        st.write("📝 NLP")

    with c3:

        st.write("🌐 Streamlit")

        st.write("📈 Plotly")

    st.divider()

    # ------------------------------------------------------

    st.subheader("Project Highlights")

    st.success("""

✔ Customer Support Ticket Analysis

✔ Business KPI Dashboard

✔ Interactive Visualizations

✔ NLP Text Processing

✔ TF-IDF Feature Engineering

✔ Random Forest Classification

✔ Automatic Ticket Categorization

✔ Streamlit Web Application

""")

# ==========================================================
# DASHBOARD
# ==========================================================

elif page == "Dashboard":

    st.header("📊 Dashboard")

    st.info("Dashboard page will be added in Part 2.")

# ==========================================================
# BUSINESS INSIGHTS
# ==========================================================

elif page == "Business Insights":

    st.header("📈 Business Insights")

    st.info("Business Insights page will be added in Part 3.")

# ==========================================================
# AI TICKET CLASSIFIER
# ==========================================================

elif page == "AI Ticket Classifier":

    st.header("🤖 AI Ticket Classifier")

    st.info("AI Ticket Classifier will be added in Part 4.")

# ==========================================================
# DATASET EXPLORER
# ==========================================================

elif page == "Dataset Explorer":

    st.header("🔍 Dataset Explorer")

    st.info("Dataset Explorer will be added in Part 5.")

# ==========================================================
# ABOUT PROJECT
# ==========================================================

elif page == "About Project":

    st.header("ℹ️ About Project")

    st.write("""
### AI Customer Support Log Analyzer

This project was developed as part of a **PGDBA Business Analytics Capstone Project**.

**Objective**

To analyze customer support tickets, generate business insights,
and automatically classify customer complaints using NLP
and Machine Learning.

**Machine Learning Model**

- Random Forest Classifier

**Tools Used**

- Python
- Pandas
- Scikit-learn
- NLP
- Plotly
- Streamlit

""")

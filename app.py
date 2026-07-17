# ==========================================================
# AI CUSTOMER SUPPORT LOG ANALYZER
# Streamlit Application
# ==========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px

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
    return pd.read_csv("customer_support_cleaned.xls")

df = load_data()

dashboard_df = pd.read_csv("dashboard_data.xls")
business_df = pd.read_csv("business_summary.xls")
product_df = pd.read_csv("product_summary.xls")

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

# ==========================================================
# DASHBOARD
# ==========================================================
if page == "Dashboard":

    st.title("📊 Customer Support Dashboard")
    st.markdown("Business overview of customer support operations")

    # ------------------------------------------------------
    # KPI CARDS
    # ------------------------------------------------------

    total_tickets = len(df)
    open_tickets = (df["Ticket Status"] == "Open").sum()
    closed_tickets = (df["Ticket Status"] == "Closed").sum()

    avg_rating = df["Customer Satisfaction Rating"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Tickets",
        f"{total_tickets:,}"
    )

    col2.metric(
        "Open Tickets",
        f"{open_tickets:,}"
    )

    col3.metric(
        "Closed Tickets",
        f"{closed_tickets:,}"
    )

    col4.metric(
        "Avg Customer Rating",
        avg_rating = df["Customer Satisfaction Rating"].mean()

if pd.isna(avg_rating):
    avg_rating = 0
    )
    st.divider()

#Ticket status
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Ticket Status")

        status_counts = (
            df["Ticket Status"]
            .value_counts()
        )

        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Ticket Status Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

#ticket type
    with col2:

        st.subheader("Ticket Type")

        type_counts = (
            df["Ticket Type"]
            .value_counts()
        )

        fig = px.bar(
            x=type_counts.index,
            y=type_counts.values,
            labels={
                "x":"Ticket Type",
                "y":"Count"
            },
            title="Tickets by Category"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

#priority & support channel
    st.divider()

    col3, col4 = st.columns(2)

#ticket priority
    with col3:

        st.subheader("Ticket Priority")

        priority_counts = (
            df["Ticket Priority"]
            .value_counts()
        )

        fig = px.bar(
            priority_counts,
            x=priority_counts.index,
            y=priority_counts.values,
            color=priority_counts.index,
            labels={
                "x":"Priority",
                "y":"Tickets"
            },
            title="Priority Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

#support channel
    with col4:

        st.subheader("Support Channel")

        channel_counts = (
            df["Ticket Channel"]
            .value_counts()
        )

        fig = px.pie(
            values=channel_counts.values,
            names=channel_counts.index,
            title="Support Channels"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

#display dataset
    st.divider()

    st.subheader("Customer Support Dataset")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


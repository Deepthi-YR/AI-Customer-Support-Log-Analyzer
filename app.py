# ==========================================================
# AI CUSTOMER SUPPORT LOG ANALYZER
# Streamlit Application
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Customer Support Log Analyzer",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background-color:#F8F9FA;
}

h1,h2,h3{
    color:#003366;
}

div[data-testid="metric-container"]{
    background-color:white;
    border-radius:12px;
    padding:15px;
    box-shadow:0 2px 8px rgba(0,0,0,0.08);
}

.sidebar .sidebar-content{
    background:#003366;
}

hr{
    margin-top:0;
    margin-bottom:1rem;
}

</style>
""", unsafe_allow_html=True)

st.title("🎧 AI Customer Support Log Analyzer")

st.markdown("""
Analyze customer support tickets using Machine Learning,
visualize operational KPIs, and predict ticket categories
from customer complaints.
""")

st.markdown("---")

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_xls("dashboard_data.xls")

    return df


@st.cache_resource
def load_models():

    model = joblib.load("best_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    encoder = joblib.load("label_encoder.pkl")

    return model, vectorizer, encoder


df = load_data()

model, tfidf, label_encoder = load_models()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Page",

    [
        "Dashboard",
        "AI Ticket Classifier",
        "Business Insights",
        "About Project"
    ]

)

# ==========================================================
# DASHBOARD
# ==========================================================

if page == "Dashboard":

    st.header("📊 Dashboard")

    st.write("Overview of Customer Support Performance")

    st.markdown("---")


    total_tickets = len(df)

    open_tickets = len(df[df["Ticket Status"]=="Open"])

    closed_tickets = len(df[df["Ticket Status"]=="Closed"])

    avg_rating = round(
        df["Customer Satisfaction Rating"].mean(),
        2
    )

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Total Tickets",
        f"{total_tickets:,}"
    )

    c2.metric(
        "Open Tickets",
        f"{open_tickets:,}"
    )

    c3.metric(
        "Closed Tickets",
        f"{closed_tickets:,}"
    )

    c4.metric(
        "Average Rating",
        avg_rating
    )

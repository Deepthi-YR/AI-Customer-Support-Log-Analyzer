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
    label_encoder = joblib.load("label_encoder.pkl")

    return model, vectorizer, label_encoder


model, vectorizer, label_encoder = load_model()

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

# ==========================================================
# DASHBOARD
# ==========================================================

elif page == "📊 Dashboard":

    st.title("📊 Customer Support Dashboard")
    st.markdown("### Business Performance Overview")

    # ------------------------------------------------------
    # KPI CARDS
    # ------------------------------------------------------

    total_tickets = len(df)

    open_tickets = (
        df["Ticket Status"]
        .astype(str)
        .str.lower()
        .eq("open")
        .sum()
    )

    closed_tickets = (
        df["Ticket Status"]
        .astype(str)
        .str.lower()
        .eq("closed")
        .sum()
    )

    avg_rating = df["Customer Satisfaction Rating"].fillna(0).mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Tickets", f"{total_tickets:,}")
    col2.metric("Open Tickets", open_tickets)
    col3.metric("Closed Tickets", closed_tickets)
    col4.metric("Avg Rating", f"{avg_rating:.2f}")

    st.markdown("---")

    # ------------------------------------------------------
    # FIRST ROW
    # ------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Ticket Status")

        status = (
            df["Ticket Status"]
            .value_counts()
            .reset_index()
        )

        status.columns = ["Status", "Count"]

        fig = px.pie(
            status,
            names="Status",
            values="Count",
            hole=0.45
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Ticket Categories")

        ticket_type = (
            df["Ticket Type"]
            .value_counts()
            .reset_index()
        )

        ticket_type.columns = ["Type", "Count"]

        fig = px.bar(
            ticket_type,
            x="Type",
            y="Count",
            text="Count",
            color="Type"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------
    # SECOND ROW
    # ------------------------------------------------------

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Priority Distribution")

        priority = (
            df["Ticket Priority"]
            .value_counts()
            .reset_index()
        )

        priority.columns = ["Priority", "Count"]

        fig = px.bar(
            priority,
            x="Priority",
            y="Count",
            color="Priority",
            text="Count"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col4:

        st.subheader("Support Channels")

        channel = (
            df["Ticket Channel"]
            .value_counts()
            .reset_index()
        )

        channel.columns = ["Channel", "Count"]

        fig = px.pie(
            channel,
            names="Channel",
            values="Count",
            hole=0.45
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------
    # THIRD ROW
    # ------------------------------------------------------

    if "Date of Purchase" in df.columns:

        st.subheader("Ticket Trend")

        trend = df.copy()

        trend["Date of Purchase"] = pd.to_datetime(
            trend["Date of Purchase"],
            errors="coerce"
        )

        trend = (
            trend
            .groupby("Date of Purchase")
            .size()
            .reset_index(name="Tickets")
            .sort_values("Date of Purchase")
        )

        fig = px.line(
            trend,
            x="Date of Purchase",
            y="Tickets",
            markers=True
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------
    # DATA PREVIEW
    # ------------------------------------------------------

    st.subheader("Recent Customer Support Tickets")

    st.dataframe(
        df.head(15),
        use_container_width=True
    )

# ==========================================================
# BUSINESS INSIGHTS
# ==========================================================

elif page == "📈 Business Insights":
    
    st.title("📈 Business Insights")
    st.markdown("### Operational Performance & Customer Insights")

    # ------------------------------------------------------
    # KPI SUMMARY
    # ------------------------------------------------------

    total_tickets = len(df)
    
    avg_rating = df["Customer Satisfaction Rating"].dropna().mean()
    
    top_issue = df["Ticket Type"].mode()[0]
    
    top_product = df["Product Purchased"].mode()[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Tickets", total_tickets)
    
    col2.metric(
        "Avg Rating",
        f"{avg_rating:.2f}/5" if pd.notna(avg_rating) else "N/A"
    )
    
    col3.metric("Top Issue", top_issue)
    
    col4.metric("Most Reported Product", top_product)
    
    st.markdown("---")


    # ------------------------------------------------------
    # CUSTOMER SATISFACTION
    # ------------------------------------------------------

    if df["Customer Satisfaction Rating"].notna().sum() > 0:

        st.subheader("Customer Satisfaction Ratings")

        rating = (
            df["Customer Satisfaction Rating"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        rating.columns = ["Rating", "Count"]

        fig = px.bar(
            rating,
            x="Rating",
            y="Count",
            text="Count",
            color="Rating"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------
    # PRODUCT-WISE ISSUES
    # ------------------------------------------------------

    st.subheader("Top Products with Support Tickets")

    product = (
        df["Product Purchased"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    product.columns = ["Product", "Tickets"]

    fig = px.bar(
        product,
        x="Product",
        y="Tickets",
        text="Tickets",
        color="Tickets"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")


    # ------------------------------------------------------
    # TICKET PRIORITY
    # ------------------------------------------------------
    
    st.subheader("Ticket Priority Distribution")
    
    priority = (
        df["Ticket Priority"]
        .value_counts()
        .reset_index()
    )
    
    priority.columns = ["Priority", "Tickets"]
    
    fig = px.bar(
        priority,
        x="Priority",
        y="Tickets",
        text="Tickets",
        color="Priority"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")

    # ------------------------------------------------------
    # ROOT CAUSE ANALYSIS
    # ------------------------------------------------------

    st.subheader("Root Cause Analysis")

    issue = (
        df["Ticket Type"]
        .value_counts()
        .reset_index()
    )

    issue.columns = ["Issue Type", "Count"]

    fig = px.pie(
        issue,
        names="Issue Type",
        values="Count",
        hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------
    # TICKET TREND
    # ------------------------------------------------------
    
    st.subheader("Ticket Trend")
    
    trend = df.copy()
    
    trend["Date of Purchase"] = pd.to_datetime(
        trend["Date of Purchase"],
        errors="coerce"
    )
    
    trend = (
        trend.groupby("Date of Purchase")
        .size()
        .reset_index(name="Tickets")
    )
    
    fig = px.line(
        trend,
        x="Date of Purchase",
        y="Tickets",
        markers=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")

    # ------------------------------------------------------
    # BUSINESS RECOMMENDATIONS
    # ------------------------------------------------------

    st.subheader("📌 Key Business Recommendations")
    
    st.success("""
    ✔ Technical issues represent the largest proportion of customer support tickets.
    
    ✔ Prioritize High Priority tickets for faster resolution.
    
    ✔ Focus quality improvement efforts on products receiving the highest complaint volumes.
    
    ✔ Monitor Customer Satisfaction Ratings regularly to improve service quality.
    
    ✔ Deploy AI-based ticket classification to automate ticket routing and reduce manual effort.
    
    ✔ Continuously analyze recurring ticket categories to identify product and service improvement opportunities.
    """)


# ==========================================================
# AI TICKET PREDICTOR
# ==========================================================

elif page == "🤖 AI Ticket Predictor":

    st.title("🤖 AI Ticket Category Predictor")
    st.markdown(
        "Predict the category of a customer support ticket using the trained XGBoost model."
    )

    st.markdown("---")

    user_text = st.text_area(
        "Enter Customer Support Ticket",
        height=180,
        placeholder="Example:\nI purchased a laptop last week and it keeps shutting down automatically."
    )

    if st.button("Predict Ticket Category"):

        if user_text.strip() == "":
            st.warning("⚠ Please enter a customer support ticket.")

        else:

            try:

                # Convert text using saved TF-IDF vectorizer
                text_vector = vectorizer.transform([user_text])

                # Predict category
                prediction = model.predict(text_vector)[0]

                st.success(f"✅ Predicted Ticket Category: **{prediction}**")

                # ---------------------------------------------------
                # Prediction Confidence & Top Predictions
                # ---------------------------------------------------

                if hasattr(model, "predict_proba"):

                    prediction_num = model.predict(text_vector)[0]

                    prediction = label_encoder.inverse_transform([prediction_num])[0]
                    
                    st.success(f"✅ Predicted Ticket Category: **{prediction}**")

                    confidence = np.max(probability) * 100

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.2f}%"
                    )

                    top3 = np.argsort(probability)[::-1][:3]

                    decoded_labels = label_encoder.inverse_transform(model.classes_[top3])

                    results = pd.DataFrame({
                        "Category": decoded_labels,
                        "Confidence (%)": np.round(probability[top3] * 100, 2)
                    })

                    st.subheader("Top 3 Predictions")
                    st.dataframe(results, use_container_width=True)

                st.markdown("---")

                # ---------------------------------------------------
                # Suggested Action
                # ---------------------------------------------------

                st.subheader("Suggested Action")

                actions = {
                    "Technical Issue":
                        "🔧 Assign to Technical Support Team. Investigate the issue immediately.",

                    "Billing Inquiry":
                        "💳 Forward to Billing Team for payment verification.",

                    "Refund Request":
                        "💰 Verify refund eligibility and initiate refund process.",

                    "Product Inquiry":
                        "📦 Share product information or connect with Sales Team.",

                    "Account Access":
                        "🔐 Help customer reset password or verify account.",

                    "Cancellation Request":
                        "❌ Process cancellation request and notify the customer.",

                    "Shipping Issue":
                        "🚚 Contact Logistics Team and provide shipment update.",

                    "General Inquiry":
                        "📞 Route to General Customer Support Team."
                }

                action = actions.get(
                    prediction,
                    "📞 Route to General Customer Support Team."
                )

                st.success(action)
                st.write("Model Classes:", model.classes_)
                st.write("Label Encoder Classes:", label_encoder.classes_)

            except Exception as e:

                st.error(f"Prediction Error: {e}")

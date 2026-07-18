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
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from textblob import TextBlob

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
    # ---------------------------------------
    # Sentiment Analysis using TextBlob
    # ---------------------------------------
    
    def get_sentiment(text):
        text = str(text)
    
        polarity = TextBlob(text).sentiment.polarity
    
        if polarity > 0:
            return "Positive"
        elif polarity < 0:
            return "Negative"
        else:
            return "Neutral"
    
    df["Sentiment"] = (
        df["Ticket Subject"].fillna("") + " " +
        df["Ticket Description"].fillna("")
    ).apply(get_sentiment)

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
# HOME
# ==========================================================

if page == "🏠 Home":

    st.title("🤖 AI Customer Support Log Analyzer")

    st.markdown("""
Welcome to the **AI Customer Support Log Analyzer**, an intelligent analytics application designed to help organizations analyze customer support tickets, understand customer sentiment, identify recurring issues, and automate ticket categorization using Machine Learning.

This application combines **Natural Language Processing (NLP)**, **XGBoost Machine Learning**, and **Business Intelligence Dashboards** to improve customer support operations and decision-making.
""")

    st.divider()

    # ------------------------------------------------------
    # PROJECT HIGHLIGHTS
    # ------------------------------------------------------

    st.subheader("🚀 Project Highlights")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📩 Tickets", f"{len(df):,}")

    with col2:
        st.metric("🎯 Accuracy", "99.94%")

    with col3:
        st.metric("🧠 ML Model", "XGBoost")

    with col4:
        st.metric("📊 Cross Validation", "99.97%")

    st.divider()

    # ------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------

    st.subheader("✨ Key Features")

    st.markdown("""
✅ Customer Support Dashboard

✅ AI Ticket Category Prediction

✅ Customer Sentiment Analysis

✅ Root Cause Analysis

✅ Business Insights & Recommendations

✅ Interactive Data Explorer

✅ Downloadable Reports
""")

    st.divider()

    # ------------------------------------------------------
    # APPLICATION WORKFLOW
    # ------------------------------------------------------

    st.subheader("🔄 Application Workflow")

    st.markdown("""
1️⃣ Customer support tickets are loaded.

⬇️

2️⃣ Text is cleaned and transformed using **TF-IDF**.

⬇️

3️⃣ The trained **XGBoost** model predicts the ticket category.

⬇️

4️⃣ Customer sentiment is analyzed using **TextBlob**.

⬇️

5️⃣ Dashboards generate KPIs, trends, and business insights.

⬇️

6️⃣ AI recommendations assist support teams in faster decision-making.
""")

    st.divider()

    # ------------------------------------------------------
    # PROJECT MODULES
    # ------------------------------------------------------

    st.subheader("📂 Modules Available")

    modules = pd.DataFrame({
        "Module": [
            "📊 Dashboard",
            "📈 Business Insights",
            "🤖 AI Ticket Predictor",
            "😊 Sentiment Analysis",
            "📂 Data Explorer",
            "ℹ️ About"
        ],
        "Purpose": [
            "Monitor customer support KPIs",
            "Generate operational insights",
            "Predict ticket categories using XGBoost",
            "Analyze customer emotions",
            "Explore and filter the dataset",
            "Project overview and methodology"
        ]
    })

    st.dataframe(modules, use_container_width=True, hide_index=True)

    st.divider()

    # ------------------------------------------------------
    # BUSINESS VALUE
    # ------------------------------------------------------

    st.subheader("💼 Business Value")

    st.success("""
✔ Reduces manual ticket classification.

✔ Improves customer support efficiency.

✔ Identifies recurring customer issues.

✔ Supports data-driven decision-making.

✔ Enhances customer satisfaction through faster ticket routing.
""")

    st.divider()

    st.info(
        "👈 Use the navigation menu on the left to explore the different modules of the application."
    )

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
    # AI Generated Insights
    # ------------------------------------------------------
    
    total_tickets = len(df)
    avg_rating = df["Customer Satisfaction Rating"].dropna().mean()
    top_issue = df["Ticket Type"].mode()[0]
    top_product = df["Product Purchased"].mode()[0]

    st.info(f"""
    ### 🤖 AI Summary
    
    These insights are automatically generated from the customer support dataset using machine learning and business analytics.
    """)

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
    
    st.subheader("📌 AI Business Recommendations")
    
    recommendations = []
    
    if top_issue == "Technical issue":
        recommendations.append(
            "🔧 Increase Technical Support staffing to reduce resolution time."
        )
    
    if avg_rating < 3:
        recommendations.append(
            "⭐ Customer satisfaction is below target. Improve response quality and reduce response time."
        )
    else:
        recommendations.append(
            "😊 Customer satisfaction is healthy. Maintain current service standards."
        )
    
    recommendations.append(
        "📊 Monitor recurring ticket categories to identify product improvements."
    )
    
    recommendations.append(
        "🤖 Continue using AI-based ticket classification to automate ticket routing."
    )
    
    recommendations.append(
        "🚨 Review High Priority tickets daily to improve SLA compliance."
    )
    
    for rec in recommendations:
        st.success(rec)
    
    st.download_button(
        label="📥 Download Business Insights",
        data=df.to_csv(index=False),
        file_name="business_insights.csv",
        mime="text/csv"
    )

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
                prediction_num = model.predict(text_vector)[0]

                prediction = label_encoder.inverse_transform([prediction_num])[0]
                
                st.success(f"✅ Predicted Ticket Category: **{prediction}**")

                st.write("Prediction Number:", prediction_num)
                st.write("Prediction Label:", prediction)

                # ---------------------------------------------------
                # Prediction Confidence & Top Predictions
                # ---------------------------------------------------

                if hasattr(model, "predict_proba"):

                    # Get prediction probabilities
                    probability = model.predict_proba(text_vector)[0]
                
                    # Confidence Score
                    confidence = np.max(probability) * 100
                
                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.2f}%"
                    )
                
                    # Top 3 Predictions
                    top3 = np.argsort(probability)[::-1][:3]
                
                    decoded_labels = label_encoder.inverse_transform(model.classes_[top3])
                
                    results = pd.DataFrame({
                        "Category": decoded_labels,
                        "Confidence (%)": np.round(probability[top3] * 100, 2)
                    })
                
                    st.subheader("Top 3 Predictions")
                    st.dataframe(results, use_container_width=True)
                
                # ---------------------------------------------------
                # Suggested Action
                # ---------------------------------------------------

                st.subheader("Suggested Action")

                actions = {
                    "Technical issue":
                        "🔧 Assign to Technical Support Team. Investigate the issue immediately.",
                
                    "Billing inquiry":
                        "💳 Forward to Billing Team for payment verification.",
                
                    "Refund request":
                        "💰 Verify refund eligibility and initiate the refund process.",
                
                    "Product inquiry":
                        "📦 Share product details or connect the customer with the Sales Team.",
                
                    "Cancellation request":
                        "❌ Process the cancellation request and send a confirmation email."
                }
        
                action = actions.get(
                    prediction,
                    "📋 Review the ticket manually and assign it to the appropriate support team."
                )
                
                st.success(action)    

            except Exception as e:

                st.error(f"Prediction Error: {e}")

# ==========================================================
# SENTIMENT ANALYSIS
# ==========================================================
elif page == "😊 Sentiment Analysis":

    st.title("😊 Customer Sentiment Analysis")

    st.markdown("Analyze customer emotions from support tickets using AI sentiment analysis.")

    # -----------------------------------------
    # Sentiment Counts
    # -----------------------------------------

    sentiment_counts = df["Sentiment"].value_counts()

    positive = sentiment_counts.get("Positive", 0)
    neutral = sentiment_counts.get("Neutral", 0)
    negative = sentiment_counts.get("Negative", 0)

    total = positive + neutral + negative

    # -----------------------------------------
    # KPI Cards
    # -----------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("😊 Positive", positive)

    with c2:
        st.metric("😐 Neutral", neutral)

    with c3:
        st.metric("😠 Negative", negative)

    st.divider()

    # -----------------------------------------
    # Pie Chart
    # -----------------------------------------

    fig1 = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Overall Customer Sentiment Distribution",
        hole=0.4
    )

    st.plotly_chart(fig1, use_container_width=True)

    # -----------------------------------------
    # Bar Chart
    # -----------------------------------------

    fig2 = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        color=sentiment_counts.index,
        title="Sentiment Count"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # -----------------------------------------
    # Sample Comments
    # -----------------------------------------

    st.subheader("Sample Customer Comments")

    sentiment_choice = st.selectbox(
        "Select Sentiment",
        ["Positive", "Neutral", "Negative"]
    )

    sample_comments = df[
        df["Sentiment"] == sentiment_choice
    ][["Ticket Subject", "Ticket Description"]].head(5)

    st.dataframe(sample_comments, use_container_width=True)

    st.divider()

    # -----------------------------------------
    # Business Insight
    # -----------------------------------------

    st.subheader("Business Insight")

    if positive > negative:

        st.success("""
Most customer interactions are positive, indicating customers are generally satisfied with the support experience.

Recommendation:
- Maintain current service quality.
- Continue monitoring customer feedback.
""")

    else:

        st.warning("""
Negative sentiment is relatively high.

Recommendation:
- Investigate recurring issues.
- Improve response time.
- Provide additional training to support teams.
""")

# ==========================================================
# SENTIMENT ANALYSIS
# ==========================================================
elif page == "📂 Data Explorer":

    st.title("📂 Data Explorer")

    st.markdown("Explore the customer support dataset interactively.")

    st.divider()

    # -----------------------------
    # Dataset Overview
    # -----------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()

    # -----------------------------
    # Search Ticket
    # -----------------------------

    search = st.text_input("🔍 Search Ticket Subject")

    filtered_df = df.copy()

    if search:
        filtered_df = filtered_df[
            filtered_df["Ticket Subject"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    # -----------------------------
    # Ticket Type Filter
    # -----------------------------

    ticket_filter = st.multiselect(
        "Filter by Ticket Type",
        sorted(df["Ticket Type"].unique())
    )

    if ticket_filter:
        filtered_df = filtered_df[
            filtered_df["Ticket Type"].isin(ticket_filter)
        ]

    st.dataframe(filtered_df, use_container_width=True)

    st.divider()

    st.download_button(
        "📥 Download Filtered Data",
        filtered_df.to_csv(index=False),
        "filtered_customer_support_data.csv",
        "text/csv"
    )

# ==========================================================
# ABOUT
# ==========================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About the Project")

    st.markdown("""
    ## AI Customer Support Log Analyzer

    The **AI Customer Support Log Analyzer** is a machine learning-powered application developed to help organizations analyze customer support tickets, identify common issues, understand customer sentiment, and automate ticket categorization.

    This project combines **Natural Language Processing (NLP)**, **Machine Learning**, and **Business Analytics** to improve customer support efficiency and assist decision-makers with actionable insights.
    """)

    st.divider()

    st.subheader("🎯 Project Objectives")

    st.markdown("""
    - Automatically classify customer support tickets.
    - Analyze customer sentiment.
    - Identify recurring support issues.
    - Monitor customer satisfaction.
    - Generate business insights and recommendations.
    - Support faster and more accurate ticket routing.
    """)

    st.divider()

    st.subheader("🛠 Technologies Used")

    tech = {
        "Python": "✔",
        "Pandas": "✔",
        "NumPy": "✔",
        "Scikit-learn": "✔",
        "XGBoost": "✔",
        "NLTK / TextBlob": "✔",
        "Plotly": "✔",
        "Streamlit": "✔",
        "Joblib": "✔"
    }

    st.table(pd.DataFrame({
        "Technology": tech.keys(),
        "Used": tech.values()
    }))

    st.divider()

    st.subheader("📊 Machine Learning Pipeline")

    st.markdown("""
    1. Data Cleaning & Preprocessing
    2. Text Cleaning & Lemmatization
    3. TF-IDF Feature Extraction
    4. Label Encoding
    5. Train-Test Split
    6. XGBoost Classification
    7. RandomizedSearchCV Hyperparameter Tuning
    8. 5-Fold Cross Validation
    9. AI Ticket Prediction
    """)

    st.divider()

    st.subheader("🏆 Final Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Test Accuracy", "99.94%")

    with col2:
        st.metric("Cross Validation", "99.97%")

    st.success("🏅 Final Model: XGBoost Classifier")

    st.divider()

    st.subheader("💡 Business Value")

    st.info("""
    • Reduces manual ticket classification.

    • Improves customer support efficiency.

    • Enables proactive issue identification.

    • Provides actionable business insights.

    • Enhances customer satisfaction through faster response handling.
    """)

    st.divider()

    st.caption("Developed as a PGDBA Capstone Project using Python, Machine Learning, NLP, and Streamlit.")



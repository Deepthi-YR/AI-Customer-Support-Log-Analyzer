# ==========================================================
# AI CUSTOMER SUPPORT LOG ANALYZER
# STREAMLIT APPLICATION
# PART 1A - HOME PAGE
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
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

    file_path = "customer_support_cleaned.xls"

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

    st.write("Monitor key customer support metrics and ticket trends.")

    st.divider()

    # --------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total Tickets", total_tickets)

    k2.metric("Closed Tickets", closed_tickets)

    k3.metric("Open Tickets", open_tickets)

    k4.metric("Average Rating", avg_rating)

    st.divider()

    # --------------------------------------------------
    # FIRST ROW
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Ticket Status Distribution")

        status_df = (
            df["Ticket Status"]
            .value_counts()
            .reset_index()
        )

        status_df.columns = ["Status", "Count"]

        fig = px.pie(
            status_df,
            names="Status",
            values="Count",
            hole=0.45
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Ticket Type Distribution")

        ticket_df = (
            df["Ticket Type"]
            .value_counts()
            .reset_index()
        )

        ticket_df.columns = ["Ticket Type", "Count"]

        fig = px.bar(
            ticket_df,
            x="Ticket Type",
            y="Count",
            color="Ticket Type",
            text="Count"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # SECOND ROW
    # --------------------------------------------------

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Ticket Priority")

        priority_df = (
            df["Ticket Priority"]
            .value_counts()
            .reset_index()
        )

        priority_df.columns = ["Priority", "Count"]

        fig = px.bar(
            priority_df,
            x="Priority",
            y="Count",
            color="Priority",
            text="Count"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col4:

        st.subheader("Ticket Channel")

        channel_df = (
            df["Ticket Channel"]
            .value_counts()
            .reset_index()
        )

        channel_df.columns = ["Channel", "Count"]

        fig = px.pie(
            channel_df,
            names="Channel",
            values="Count",
            hole=0.45
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # MONTHLY TREND
    # --------------------------------------------------

    st.subheader("Monthly Ticket Trend")

    monthly_df = df.copy()

    monthly_df["Month"] = (
        monthly_df["Date of Purchase"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        monthly_df
        .groupby("Month")
        .size()
        .reset_index(name="Tickets")
    )

    fig = px.line(
        monthly,
        x="Month",
        y="Tickets",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# BUSINESS INSIGHTS
# ==========================================================

elif page == "Business Insights":

    st.header("📈 Business Insights")

    st.write("Analyze customer support trends and derive actionable business insights.")

    st.divider()

    # --------------------------------------------------
    # FIRST ROW
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Ticket Priority")

        priority_df = (
            df["Ticket Priority"]
            .value_counts()
            .reset_index()
        )

        priority_df.columns = ["Priority", "Count"]

        fig = px.bar(
            priority_df,
            x="Priority",
            y="Count",
            color="Priority",
            text="Count"
        )

        fig.update_traces(textposition="outside")

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Ticket Channel")

        channel_df = (
            df["Ticket Channel"]
            .value_counts()
            .reset_index()
        )

        channel_df.columns = ["Channel", "Count"]

        fig = px.pie(
            channel_df,
            names="Channel",
            values="Count",
            hole=0.45
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # SECOND ROW
    # --------------------------------------------------

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Top 10 Products with Support Tickets")

        product_df = (
            df["Product Purchased"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        product_df.columns = ["Product", "Count"]

        fig = px.bar(
            product_df,
            x="Product",
            y="Count",
            color="Count",
            text="Count"
        )

        fig.update_layout(xaxis_tickangle=-35)

        st.plotly_chart(fig, use_container_width=True)

    with col4:

        st.subheader("Customer Satisfaction Ratings")

        rating_df = (
            df["Customer Satisfaction Rating"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        rating_df.columns = ["Rating", "Count"]

        fig = px.bar(
            rating_df,
            x="Rating",
            y="Count",
            color="Rating",
            text="Count"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # THIRD ROW
    # --------------------------------------------------

    col5, col6 = st.columns(2)

    with col5:

        st.subheader("Customer Gender Distribution")

        gender_df = (
            df["Customer Gender"]
            .value_counts()
            .reset_index()
        )

        gender_df.columns = ["Gender", "Count"]

        fig = px.pie(
            gender_df,
            names="Gender",
            values="Count",
            hole=0.45
        )

        st.plotly_chart(fig, use_container_width=True)

    with col6:

        st.subheader("Top 10 Ticket Categories")

        ticket_df = (
            df["Ticket Type"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        ticket_df.columns = ["Ticket Type", "Count"]

        fig = px.bar(
            ticket_df,
            x="Ticket Type",
            y="Count",
            color="Count",
            text="Count"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # BUSINESS RECOMMENDATIONS
    # --------------------------------------------------

    st.subheader("📌 Key Business Insights")

    top_priority = df["Ticket Priority"].mode()[0]
    top_channel = df["Ticket Channel"].mode()[0]
    top_product = df["Product Purchased"].mode()[0]
    top_ticket = df["Ticket Type"].mode()[0]

    st.success(f"""
### Summary

• **Most common ticket priority:** {top_priority}

• **Most frequently used support channel:** {top_channel}

• **Product receiving the highest number of support tickets:** {top_product}

• **Most common ticket category:** {top_ticket}

### Recommendations

✅ Prioritize resources for **{top_ticket}** tickets.

✅ Improve customer support through the **{top_channel}** channel.

✅ Review quality and customer feedback for **{top_product}**.

✅ Monitor **{top_priority}** priority tickets to improve response efficiency.
""")

# ==========================================================
# AI TICKET CLASSIFIER
# ==========================================================

elif page == "AI Ticket Classifier":

    st.header("🤖 AI Ticket Category Predictor")

    st.write("""
Enter a customer complaint below.

The trained **Random Forest Machine Learning model**
will predict the most appropriate ticket category.
""")

    st.divider()

    # --------------------------------------------------
    # USER INPUT
    # --------------------------------------------------

    user_text = st.text_area(
        "Enter Customer Complaint",
        height=180,
        placeholder="Example: My laptop battery drains within 30 minutes after charging."
    )

    # --------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------

    if st.button("Predict Ticket Category"):

        if user_text.strip() == "":

            st.warning("Please enter a customer complaint.")

        elif model is None:

            st.error("Model files are not available.")

        else:

            # -----------------------------
            # Prediction
            # -----------------------------

            text_vector = vectorizer.transform([user_text])

            prediction = model.predict(text_vector)[0]

            category = label_encoder.inverse_transform([prediction])[0]

            st.success(f"### 🎯 Predicted Ticket Category: **{category}**")

            st.divider()

            # -----------------------------
            # Confidence
            # -----------------------------

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(text_vector)[0]

                confidence = probabilities.max() * 100

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

                class_names = label_encoder.inverse_transform(model.classes_)

                confidence_df = pd.DataFrame({

                    "Ticket Category": class_names,

                    "Confidence (%)":
                    (probabilities * 100).round(2)

                })

                confidence_df = confidence_df.sort_values(

                    "Confidence (%)",

                    ascending=False

                )

                st.subheader("Confidence for All Categories")

                st.dataframe(

                    confidence_df,

                    hide_index=True,

                    use_container_width=True

                )

            st.divider()

            # -----------------------------
            # Keywords
            # -----------------------------

            st.subheader("Detected Keywords")

            keywords = [

                word.strip(".,!?")

                for word in user_text.lower().split()

                if len(word) > 2

            ]

            st.write(", ".join(keywords[:10]))

            st.divider()

            # -----------------------------
            # Interpretation
            # -----------------------------

            st.subheader("Prediction Interpretation")

            st.info(f"""

**Predicted Category**

{category}

The complaint was cleaned and transformed using
**TF-IDF Vectorization**.

The Random Forest classifier compared the complaint
with previously learned customer support tickets.

Based on similar complaints, the ticket is classified
as **{category}**.

Prediction Confidence:

**{confidence:.2f}%**

""")

    st.divider()

    # --------------------------------------------------
    # SAMPLE COMPLAINTS
    # --------------------------------------------------

    st.subheader("Sample Complaints")

    st.markdown("""

**Billing Issue**

• I was charged twice for my order.

---

**Delivery Issue**

• My package has not arrived yet.

---

**Technical Support**

• The application crashes whenever I log in.

---

**Product Issue**

• My laptop screen keeps flickering.

---

**Refund Request**

• I cancelled my order but haven't received my refund.

""")

# ==========================================================
# DATASET EXPLORER
# ==========================================================

elif page == "Dataset Explorer":

    st.header("🔍 Dataset Explorer")

    st.write("Explore, search and filter customer support tickets.")

    st.divider()

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------

    search_text = st.text_input(
        "🔎 Search Ticket Description",
        placeholder="Enter keyword..."
    )

    filtered_df = df.copy()

    if search_text:

        filtered_df = filtered_df[
            filtered_df["Ticket Description"]
            .astype(str)
            .str.contains(search_text, case=False, na=False)
        ]

    # --------------------------------------------------
    # FILTERS
    # --------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        status = st.selectbox(
            "Ticket Status",
            ["All"] + sorted(df["Ticket Status"].dropna().unique().tolist())
        )

    with col2:

        priority = st.selectbox(
            "Ticket Priority",
            ["All"] + sorted(df["Ticket Priority"].dropna().unique().tolist())
        )

    with col3:

        channel = st.selectbox(
            "Ticket Channel",
            ["All"] + sorted(df["Ticket Channel"].dropna().unique().tolist())
        )

    if status != "All":

        filtered_df = filtered_df[
            filtered_df["Ticket Status"] == status
        ]

    if priority != "All":

        filtered_df = filtered_df[
            filtered_df["Ticket Priority"] == priority
        ]

    if channel != "All":

        filtered_df = filtered_df[
            filtered_df["Ticket Channel"] == channel
        ]

    st.divider()

    # --------------------------------------------------
    # DATASET
    # --------------------------------------------------

    st.subheader("Filtered Dataset")

    st.write(f"Showing **{len(filtered_df)}** records")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=450
    )

    # --------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Filtered Data",
        csv,
        "filtered_customer_support_data.csv",
        "text/csv"
    )

    st.divider()

    # --------------------------------------------------
    # DATA SUMMARY
    # --------------------------------------------------

    st.subheader("Dataset Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", filtered_df.shape[0])

    c2.metric("Columns", filtered_df.shape[1])

    c3.metric("Missing Values", filtered_df.isna().sum().sum())

# ==========================================================
# ABOUT PROJECT
# ==========================================================

elif page == "About Project":

    st.header("ℹ️ About Project")

    st.write("""
## 🎧 AI Customer Support Log Analyzer

This application was developed as part of the **PGDBA Business Analytics Capstone Project**.

It combines **Business Analytics**, **Natural Language Processing (NLP)** and **Machine Learning**
to analyze customer support tickets and automatically predict ticket categories.
""")

    st.divider()

    # --------------------------------------------------
    # PROJECT OBJECTIVE
    # --------------------------------------------------

    st.subheader("🎯 Project Objective")

    st.write("""
- Analyze customer support ticket data.
- Identify business trends and customer service patterns.
- Generate interactive dashboards for decision making.
- Automatically classify customer complaints using Machine Learning.
- Improve ticket routing and customer support efficiency.
""")

    st.divider()

    # --------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------

    st.subheader("📂 Dataset Information")

    c1, c2 = st.columns(2)

    with c1:

        st.metric("Total Tickets", len(df))

        st.metric("Features", df.shape[1])

    with c2:

        st.metric(
            "Ticket Categories",
            df["Ticket Type"].nunique()
        )

        st.metric(
            "Support Channels",
            df["Ticket Channel"].nunique()
        )

    st.divider()

    # --------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------

    st.subheader("🤖 Machine Learning Workflow")

    st.markdown("""

Customer Complaint

⬇️

Text Cleaning

⬇️

TF-IDF Vectorization

⬇️

Random Forest Classifier

⬇️

Predicted Ticket Category

""")

    st.divider()

    # --------------------------------------------------
    # TECHNOLOGIES
    # --------------------------------------------------

    st.subheader("🛠 Technologies Used")

    tech1, tech2 = st.columns(2)

    with tech1:

        st.write("🐍 Python")

        st.write("📊 Pandas")

        st.write("🤖 Scikit-learn")

        st.write("📝 Natural Language Processing")

    with tech2:

        st.write("📈 Plotly")

        st.write("🌐 Streamlit")

        st.write("💾 Joblib")

        st.write("📑 Machine Learning")

    st.divider()

    # --------------------------------------------------
    # PROJECT FEATURES
    # --------------------------------------------------

    st.subheader("✨ Application Features")

    st.success("""

✔ Dashboard Overview

✔ Business Insights

✔ AI Ticket Category Prediction

✔ Dataset Explorer

✔ Interactive Charts

✔ Download Filtered Dataset

✔ Automatic Ticket Classification

""")

    st.divider()

    # --------------------------------------------------
    # FUTURE SCOPE
    # --------------------------------------------------

    st.subheader("🚀 Future Scope")

    st.write("""

- Real-time customer ticket prediction

- Integration with CRM platforms

- Sentiment Analysis

- AI-powered ticket prioritization

- Automatic ticket assignment

- Live monitoring dashboard

""")

    st.divider()

    # --------------------------------------------------
    # DEVELOPER
    # --------------------------------------------------

    st.subheader("👩‍💻 Developed By")

    st.info("""

**Deepthi**

PGDBA (Business Analytics)

RV Institute of Management

Capstone Project 2026

""")

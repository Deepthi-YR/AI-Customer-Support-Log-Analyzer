# ==========================================================
# AI CUSTOMER SUPPORT LOG ANALYZER
# Streamlit Dashboard
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------

st.set_page_config(page_title="AI Customer Support Log Analyzer",page_icon="🎧",layout="wide")

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

@st.cache_data
def load_data():
    file_path = "customer_support_cleaned.csv"

    if not os.path.exists(file_path):
        st.error(f"Dataset not found: {file_path}")
        st.stop()

    return pd.read_csv(file_path)
    
# ----------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------

model = joblib.load("best_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ----------------------------------------------------------
# TITLE
# ----------------------------------------------------------

st.title("🎧 AI Customer Support Log Analyzer")
st.markdown("""
Analyze customer support tickets, visualize business insights,
and automatically classify new customer complaints using Machine Learning.
""")

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page",["Dashboard","Business Insights","AI Ticket Classifier"])


# ==========================================================
# DASHBOARD PAGE
# ==========================================================

if page == "Dashboard":

    st.header("📊 Dashboard Overview")

    # -----------------------------
    # KPI CALCULATIONS
    # -----------------------------

    total_tickets = len(df)
    closed_tickets = (df["Ticket Status"].astype(str).str.lower().str.contains("closed").sum())

    open_tickets = (df["Ticket Status"].astype(str).str.lower().str.contains("open").sum())

    avg_rating = round(df["Customer Satisfaction Rating"].mean(),2)

    # -----------------------------
    # KPI CARDS
    # -----------------------------

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Tickets",total_tickets)

    kpi2.metric("Closed Tickets",closed_tickets)

    kpi3.metric("Open Tickets",open_tickets)

    kpi4.metric("Average Rating",avg_rating)

    st.divider()

    # -----------------------------
    # STATUS DISTRIBUTION
    # -----------------------------

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ticket Status Distribution")
        status_df = (df["Ticket Status"].value_counts().reset_index())

        status_df.columns = ["Status","Count"]

        fig = px.pie(status_df,names="Status",values="Count",hole=0.45)

        st.plotly_chart(fig,use_container_width=True)

    # -----------------------------
    # TICKET TYPE
    # -----------------------------

    with col2:
        st.subheader("Ticket Type Distribution")

        ticket_df = (df["Ticket Type"].value_counts().reset_index())

        ticket_df.columns = ["Ticket Type","Count"]

        fig = px.bar(ticket_df,x="Ticket Type",y="Count",color="Ticket Type")

        st.plotly_chart(fig,use_container_width=True)

    st.divider()

    # -----------------------------
    # MONTHLY TREND
    # -----------------------------

    st.subheader("Monthly Ticket Trend")

    df["Date of Purchase"] = pd.to_datetime(df["Date of Purchase"])

    df["Month"] = (df["Date of Purchase"].dt.to_period("M").astype(str))

    monthly = (df.groupby("Month").size().reset_index(name="Tickets"))

    fig = px.line(monthly,x="Month",y="Tickets",markers=True)

    st.plotly_chart(fig,use_container_width=True)

# ==========================================================
# BUSINESS INSIGHTS PAGE
# ==========================================================

elif page == "Business Insights":

    st.header("📈 Business Insights")

    # --------------------------------------------------
    # FIRST ROW
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    # Priority Analysis
    with col1:

        st.subheader("Ticket Priority")

        priority_df = (df["Ticket Priority"].value_counts().reset_index())

        priority_df.columns = ["Priority","Count"]

        fig = px.bar(priority_df,x="Priority",y="Count",color="Priority",text="Count")

        fig.update_traces(textposition="outside")

        st.plotly_chart(fig,use_container_width=True)

    # Channel Analysis
    with col2:

        st.subheader("Ticket Channel")

        channel_df = (df["Ticket Channel"].value_counts().reset_index())

        channel_df.columns = ["Channel","Count"]

        fig = px.pie(channel_df,names="Channel",values="Count",hole=0.45)

        st.plotly_chart(fig,use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # SECOND ROW
    # --------------------------------------------------

    col3, col4 = st.columns(2)

    # Top Products
    with col3:

        st.subheader("Top 10 Products with Support Tickets")

        product_df = (df["Product Purchased"].value_counts().head(10).reset_index())

        product_df.columns = ["Product","Count"]

        fig = px.bar(product_df,x="Product",y="Count",color="Count",text="Count")

        fig.update_layout(xaxis_tickangle=-30)

        st.plotly_chart(fig,use_container_width=True)

    # Customer Satisfaction
    with col4:

        st.subheader("Customer Satisfaction Ratings")

        rating_df = (df["Customer Satisfaction Rating"].value_counts().sort_index().reset_index())

        rating_df.columns = ["Rating","Count"]

        fig = px.bar(rating_df,x="Rating",y="Count",color="Rating",text="Count")

        st.plotly_chart(fig,use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # GENDER DISTRIBUTION
    # --------------------------------------------------

    st.subheader("Customer Gender Distribution")

    gender_df = (df["Customer Gender"].value_counts().reset_index())

    gender_df.columns = ["Gender","Count"]

    fig = px.pie(gender_df,names="Gender",values="Count",hole=0.45)

    st.plotly_chart(fig,use_container_width=True)

    st.success("Business insights generated successfully!")

# ==========================================================
# AI TICKET CLASSIFIER
# ==========================================================

elif page == "AI Ticket Classifier":

    st.header("🤖 AI Ticket Category Predictor")

    st.write("""
Enter a customer complaint below. This application uses the best-performing
Random Forest Classifier trained on customer support tickets to predict
the most likely ticket category.
""")

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

        else:

            # Vectorize text
            text_vector = vectorizer.transform([user_text])

            # Predict category
            prediction = model.predict(text_vector)[0]

            st.success(f"### Predicted Ticket Category: **{prediction}**")

            # --------------------------------------------------
            # CONFIDENCE SCORE
            # --------------------------------------------------

            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(text_vector)

                confidence = probability.max() * 100

                st.metric(
                    label="Prediction Confidence",
                    value=f"{confidence:.2f}%"
                )

            # --------------------------------------------------
            # INTERPRETATION
            # --------------------------------------------------

            st.subheader("Interpretation")

            st.info(
                f"""
                Based on the complaint entered, the Random Forest Classifier predicts
                that this ticket belongs to the selected category.
                This prediction helps automatically route customer tickets to the
                appropriate support team, improving response time and service efficiency.
                """)

    # --------------------------------------------------
    # SAMPLE INPUTS
    # --------------------------------------------------

    st.divider()

    st.subheader("Sample Complaints")

    st.markdown("""
**Billing Issue**
- I was charged twice for my order.

**Product Issue**
- The laptop screen is flickering continuously.

**Delivery Issue**
- My package has not been delivered even after five days.

**Technical Support**
- The application crashes every time I log in.

**Refund Request**
- I cancelled my order but haven't received my refund.
""")


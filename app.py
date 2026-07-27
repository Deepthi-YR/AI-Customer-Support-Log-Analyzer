import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib

# ============================
# Load Dataset
# ============================

@st.cache_data
def load_data():
    return pd.read_csv("dashboard_data.xls")

df = load_data()

# ============================
# Load ML Models
# ============================

@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, vectorizer, encoder

model, vectorizer, encoder = load_model()

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Customer Support Log Analyser",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.block-container{
    padding-top:1rem;
    padding-bottom:1rem;
}

.main-title{
    font-size:42px;
    font-weight:bold;
    text-align:center;
    color:white;
}

.sub-title{
    text-align:center;
    color:white;
    font-size:18px;
}

.banner{
    background:linear-gradient(90deg,#0F62FE,#6A11CB);
    padding:30px;
    border-radius:15px;
}

div[data-testid="stMetric"]{
    background-color:#f7f7f7;
    border-radius:12px;
    padding:15px;
    box-shadow:0px 3px 8px rgba(0,0,0,.15);
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="banner">

<div class="main-title">
🤖 AI Customer Support Log Analyser
</div>

<div class="sub-title">
Intelligent Complaint Classification using NLP & Machine Learning
</div>

</div>
""", unsafe_allow_html=True)

st.write("")

# ==========================================
# NAVIGATION
# ==========================================

page = st.segmented_control(
    "Navigation",
    menu = [
        "🏠 Home",
        "📊 Dashboard",
        "📂 Dataset Explorer",
        "🤖 AI Predictor",
        "📈 Model Performance",
        "ℹ About"
    ]

page = st.segmented_control(
    "Navigation",
    options=menu,
    default=menu[0]
)
st.divider()

# ==========================================
# HOME
# ==========================================

if page == "🏠 Home":

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Dataset","Consumer Complaints")
    c2.metric("Model","Logistic Regression")
    c3.metric("NLP","TF-IDF")
    c4.metric("Framework","Streamlit")

    st.write("")

    left,right = st.columns([2,1])

    with left:

        st.header("📌 Project Overview")

        st.write("""
This AI-powered application automatically classifies customer complaints using
Natural Language Processing (NLP) and Machine Learning.

### Features

- Complaint Classification
- NLP Text Processing
- Interactive Dashboard
- Data Visualization
- Model Evaluation
- GitHub Deployment
""")

    with right:

        st.info("""
### Technologies

- Python
- Pandas
- NumPy
- NLTK
- Scikit-Learn
- Streamlit
""")

# ==========================================
# DASHBOARD
# ==========================================

elif page == "📊 Dashboard":

    st.header("📊 Analytics Dashboard")

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Complaints", len(df))
    c2.metric("Categories", df["product"].nunique())
    c3.metric("Average Length", round(df["narrative"].astype(str).str.len().mean()))
    c4.metric("Missing Values", df.isna().sum().sum())

    st.divider()

    # ------------------------------
    # Charts
    # ------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Complaint Categories")

        category_counts = (
            df["product"]
            .value_counts()
            .reset_index()
        )

        category_counts.columns = ["Product", "Count"]

        fig = px.bar(
            category_counts,
            x="Product",
            y="Count",
            color="Product",
            title="Complaint Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Category Share")

        fig2 = px.pie(
            df,
            names="product",
            hole=0.5,
            title="Complaint Share"
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ------------------------------
    # Word Cloud
    # ------------------------------

    st.subheader("☁️ Most Frequent Words")

    text = " ".join(df["narrative"].astype(str))

    wc = WordCloud(
        width=900,
        height=400,
        background_color="white"
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12,5))

    ax.imshow(wc)

    ax.axis("off")

    st.pyplot(fig)

    st.divider()

    # ------------------------------
    # Data Table
    # ------------------------------

    st.subheader("Recent Complaints")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

# ==========================================
# DATASET EXPLORER
# ==========================================

elif page == "📂 Dataset Explorer":

    st.header("📂 Dataset Explorer")

    st.write("Browse, search, filter and download the complaint dataset.")

    st.divider()

    # KPI Cards
    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", len(df))
    c2.metric("Products", df["product"].nunique())
    c3.metric("Columns", len(df.columns))

    st.divider()

    # Product Filter
    products = ["All"] + sorted(df["product"].unique().tolist())

    selected_product = st.selectbox(
        "📌 Select Product",
        products
    )

    if selected_product == "All":
        filtered_df = df.copy()
    else:
        filtered_df = df[df["product"] == selected_product]

    # Search Complaint
    search = st.text_input(
        "🔍 Search Complaint Text",
        placeholder="Type any keyword..."
    )

    if search:
        filtered_df = filtered_df[
            filtered_df["narrative"]
            .astype(str)
            .str.contains(search, case=False, na=False)
        ]

    st.divider()

    st.subheader("📋 Complaint Records")

    rows = st.slider(
    "Number of Rows",
    min_value=10,
    max_value=100,
    value=20
    )
    
    st.dataframe(
        filtered_df.head(rows),
        use_container_width=True,
        height=450
    )
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=450
    )

    st.write(f"Showing **{len(filtered_df)}** records")

    st.divider()

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Filtered Data",
        data=csv,
        file_name="filtered_complaints.csv",
        mime="text/csv"
    )

# ==========================================
# AI PREDICTOR
# ==========================================

elif page == "🤖 AI Predictor":

    st.header("🤖 AI Complaint Classifier")
    st.write("Enter a customer complaint below and let the AI predict its category.")

    complaint = st.text_area(
        "✍️ Enter Customer Complaint",
        height=200,
        placeholder="Example: My credit card was charged twice and customer support is not responding."
    )

    if st.button("🔍 Predict Category", use_container_width=True):

        if complaint.strip() == "":
            st.warning("Please enter a complaint before predicting.")
        else:

            # Transform text
            text_vector = vectorizer.transform([complaint])
            
            # Prediction with confidence
            probabilities = model.predict_proba(text_vector)[0]
            
            prediction = probabilities.argmax()
            
            predicted_category = encoder.inverse_transform([prediction])[0]
            
            confidence = probabilities.max() * 100
            
            st.success("✅ Prediction Completed!")
            
            # KPI Cards
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🎯 Predicted Category", predicted_category)
            
            with col2:
                st.metric("📊 Confidence", f"{confidence:.2f}%")
            
            # Probability Chart
            prob_df = pd.DataFrame({
                "Category": encoder.classes_,
                "Probability": probabilities
            })
            
            fig = px.bar(
                prob_df,
                x="Category",
                y="Probability",
                color="Probability",
                title="Prediction Confidence",
                text_auto=".2f"
            )
            
            fig.update_layout(
                xaxis_title="Complaint Category",
                yaxis_title="Probability",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)


# ==========================================
# MODEL PERFORMANCE
# ==========================================

elif page == "📈 Model Performance":

    st.header("📈 Model Performance")

    st.write("Performance summary of the trained Machine Learning model.")

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Model", "Best Model")
    c2.metric("Vectorizer", "TF-IDF")
    c3.metric("Classes", len(encoder.classes_))
    c4.metric("Status", "Ready")

    st.divider()

    st.subheader("🏆 Model Information")

    st.info("""
**Algorithm:** Best Performing Model

**Text Vectorization:** TF-IDF

**Label Encoding:** LabelEncoder

**Deployment:** Streamlit
""")

    st.divider()

    st.subheader("📋 Supported Complaint Categories")

    category_df = pd.DataFrame({
        "Complaint Categories": encoder.classes_
    })

    st.dataframe(
        category_df,
        use_container_width=True
    )

    st.divider()

    st.success("✅ Model loaded successfully and ready for prediction.")

# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "ℹ About":

    st.header("ℹ️ About AI Customer Support Log Analyser")

    st.markdown("""
## 🎯 Project Objective

The **AI Customer Support Log Analyser** is a Machine Learning and NLP application
that automatically classifies customer complaints into predefined categories.

It helps organizations analyze customer issues quickly, reduce manual effort,
and improve customer support efficiency.
""")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🚀 Key Features")

        st.success("""
✔ Interactive Dashboard

✔ Complaint Analytics

✔ AI Complaint Classification

✔ NLP Text Processing

✔ TF-IDF Vectorization

✔ Machine Learning Prediction

✔ Model Performance Overview

✔ Professional Streamlit Interface
""")

    with col2:

        st.subheader("🛠 Technologies Used")

        st.info("""
• Python

• Streamlit

• Pandas

• NumPy

• Plotly

• Scikit-learn

• NLTK

• Joblib

• WordCloud
""")

    st.divider()

    st.subheader("📂 Machine Learning Workflow")

    workflow = """
Customer Complaint
        │
        ▼
Text Cleaning
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Machine Learning Model
        │
        ▼
Label Decoder
        │
        ▼
Predicted Category
"""

    st.code(workflow)

    st.divider()

    st.subheader("📌 Dataset")

    st.write(
        "The project uses customer complaint narratives for supervised "
        "machine learning. Each complaint is mapped to its corresponding "
        "product category, allowing the model to learn patterns and classify "
        "new complaints."
    )

    st.divider()

    st.subheader("👨‍💻 Project Summary")

    st.write("""
This project demonstrates the complete Machine Learning lifecycle:

• Data Cleaning

• Exploratory Data Analysis

• Natural Language Processing

• Feature Engineering (TF-IDF)

• Model Training

• Model Evaluation

• Streamlit Deployment
""")

    st.divider()

    st.caption("© 2026 AI Customer Support Log Analyser | Developed using Streamlit")


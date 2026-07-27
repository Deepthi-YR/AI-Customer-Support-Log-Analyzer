import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    return pd.read_csv("dashboard_data.xls")

df = load_data()

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
    options=[
        "🏠 Home",
        "📊 Dashboard",
        "📁 Data Explorer",
        "🤖 AI Predictor",
        "📈 Model Performance",
        "ℹ About"
    ],
    default="🏠 Home"
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

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Complaints", len(df))
    c2.metric("Categories", df["Category"].nunique())
    c3.metric("Average Length", round(df["Complaint"].str.len().mean()))
    c4.metric("Missing Values", df.isna().sum().sum())

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Complaint Categories")

        fig = px.bar(
            df["Category"].value_counts().reset_index(),
            x="Category",
            y="count",
            color="Category",
            title="Complaint Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Category Share")

        fig2 = px.pie(
            df,
            names="Category",
            hole=.5
        )

        st.plotly_chart(fig2, use_container_width=True)

st.divider()

st.subheader("☁ Most Frequent Words")

text = " ".join(df["Complaint"].astype(str))

wc = WordCloud(
    width=900,
    height=400,
    background_color="white"
).generate(text)

fig, ax = plt.subplots(figsize=(12,5))

ax.imshow(wc)

ax.axis("off")

st.pyplot(fig)

    #Complaint table
st.divider()

st.subheader("Recent Complaints")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# ==========================================
# DATA EXPLORER
# ==========================================

elif page == "📁 Data Explorer":

    st.header("📁 Data Explorer")

    st.info("Dataset preview and filtering will be added.")

# ==========================================
# AI PREDICTOR
# ==========================================

elif page == "🤖 AI Predictor":

    st.header("🤖 AI Predictor")

    complaint = st.text_area(
        "Enter a customer complaint",
        height=180
    )

    if st.button("Predict Category"):

        if complaint.strip()=="":
            st.warning("Please enter a complaint.")
        else:
            st.success("Prediction will appear here.")

# ==========================================
# MODEL PERFORMANCE
# ==========================================

elif page == "📈 Model Performance":

    st.header("📈 Model Performance")

    st.info("Accuracy, Confusion Matrix and Classification Report will be displayed.")

# ==========================================
# ABOUT
# ==========================================

elif page == "ℹ About":

    st.header("ℹ About Project")

    st.write("""
### AI Customer Support Log Analyser

This project uses Machine Learning and NLP techniques to automatically classify
customer complaints into predefined categories.

#### Workflow

Complaint

⬇

Text Cleaning

⬇

TF-IDF

⬇

Machine Learning

⬇

Prediction

---

Developed using:

- Python
- Streamlit
- Scikit-Learn
- NLTK
""")

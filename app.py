import streamlit as st

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Customer Support Log Analyser",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("🤖 AI Customer Support")
st.sidebar.markdown("---")

st.sidebar.info(
"""
### Navigation

Use the pages on the left.

🏠 Home

📊 Dashboard

📁 Data Explorer

🤖 AI Predictor

📈 Model Performance

ℹ About Project
"""
)

st.sidebar.markdown("---")

st.sidebar.success("Machine Learning Project")

# ---------------- HEADER ---------------- #

st.title("🤖 AI Customer Support Log Analyser")

st.markdown(
"""
### Intelligent Customer Complaint Classification using Machine Learning

This application automatically classifies customer support complaints
using Natural Language Processing (NLP) and Machine Learning.
"""
)

st.markdown("---")

# ---------------- KPI SECTION ---------------- #

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Dataset",
        value="Consumer Complaints"
    )

with col2:
    st.metric(
        label="ML Model",
        value="Logistic Regression"
    )

with col3:
    st.metric(
        label="Vectorizer",
        value="TF-IDF"
    )

with col4:
    st.metric(
        label="Framework",
        value="Streamlit"
    )

st.markdown("---")

# ---------------- PROJECT OVERVIEW ---------------- #

left, right = st.columns([2,1])

with left:

    st.subheader("📌 Project Overview")

    st.write("""

Customer support teams receive thousands of complaints every day.

Manually categorizing these complaints is:

- Time consuming
- Error-prone
- Expensive

This AI-powered system automatically predicts the complaint category using NLP and Machine Learning.

### Features

✅ Complaint Classification

✅ Data Visualization

✅ Interactive Dashboard

✅ AI Prediction

✅ Model Performance Metrics

✅ GitHub Deployable

""")

with right:

    st.subheader("🧠 Technologies")

    st.info("""
Python

Pandas

NumPy

Matplotlib

Scikit-Learn

NLTK

TF-IDF

Logistic Regression

Streamlit
""")

st.markdown("---")

# ---------------- WORKFLOW ---------------- #

st.subheader("⚙ Project Workflow")

st.write("""

Raw Complaint

⬇

Text Cleaning

⬇

Tokenization

⬇

TF-IDF Vectorization

⬇

Machine Learning Model

⬇

Predicted Complaint Category

""")

st.markdown("---")

# ---------------- PROJECT HIGHLIGHTS ---------------- #

st.subheader("🚀 Project Highlights")

c1, c2, c3 = st.columns(3)

with c1:

    st.success("""
### NLP

✔ Text Cleaning

✔ Stopword Removal

✔ Lemmatization
""")

with c2:

    st.warning("""
### Machine Learning

✔ TF-IDF

✔ Logistic Regression

✔ Model Evaluation
""")

with c3:

    st.info("""
### Dashboard

✔ Interactive

✔ Multipage

✔ Deployable
""")

st.markdown("---")

st.caption(
    "Developed using Streamlit • NLP • Machine Learning • Scikit-Learn"
)

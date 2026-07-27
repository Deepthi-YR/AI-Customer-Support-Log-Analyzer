import streamlit as st
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Download NLTK Resources
# -----------------------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# -----------------------------
# Load Saved Files
# -----------------------------
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# -----------------------------
# NLP Preprocessing
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"\d+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()

    words = [word for word in words if word not in stop_words]

    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Customer Support Log Analyser",
    page_icon="🤖",
    layout="wide"
)

# =============================
# Sidebar
# =============================

st.sidebar.title("🤖 AI Customer Support Log Analyser")

st.sidebar.markdown("---")

st.sidebar.header("📌 About Project")

st.sidebar.write(
"""
This application uses **Natural Language Processing (NLP)** and
**Machine Learning** to automatically classify customer complaints
into the appropriate product category.

The system helps organizations understand customer issues
and route complaints efficiently.
"""
)

st.sidebar.markdown("---")

st.sidebar.header("🧠 Machine Learning Model")

st.sidebar.success("✔ Logistic Regression")

st.sidebar.write("**Model Accuracy:** 84.71%")

st.sidebar.markdown("---")

st.sidebar.header("🛠 Technologies Used")

st.sidebar.write("""
- Python
- Streamlit
- Scikit-learn
- NLP
- TF-IDF Vectorizer
- Logistic Regression
- NLTK
- Pandas
""")

st.sidebar.markdown("---")

st.sidebar.header("📂 Dataset")

st.sidebar.write("""
Consumer Complaints Dataset for NLP
""")

st.sidebar.markdown("---")

st.sidebar.header("💡 Sample Complaint")

st.sidebar.info(
"""
I have been charged twice for my credit card bill and customer support is not responding.
"""
)

st.sidebar.markdown("---")

st.sidebar.caption("Developed as an AI/ML Academic Project")

# =============================
# Main Page
# =============================

st.title("🤖 AI Customer Support Log Analyser")

st.markdown(
"""
Welcome!

This application predicts the **Product Category** of a customer complaint
using **Machine Learning** and **Natural Language Processing (NLP)**.

Simply enter a customer complaint below and click **Predict Category**.
"""
)

st.markdown("---")

user_input = st.text_area(
    "📝 Enter Customer Complaint",
    placeholder="Type the customer complaint here...",
    height=220
)

predict = st.button("🚀 Predict Category")

if predict:

    if user_input.strip() == "":

        st.warning("⚠ Please enter a customer complaint.")

    else:

        clean_text = preprocess_text(user_input)

        vector = vectorizer.transform([clean_text])

        prediction = model.predict(vector)

        category = label_encoder.inverse_transform(prediction)

        st.success("✅ Prediction Completed Successfully!")

        st.markdown("## 📌 Predicted Product Category")

        st.info(category[0])

st.markdown("---")

st.caption(
"""
AI Customer Support Log Analyser | Machine Learning & NLP Project
"""
)

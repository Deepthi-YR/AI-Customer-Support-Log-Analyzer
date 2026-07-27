import streamlit as st
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load saved model files
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# Initialize preprocessing tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Text preprocessing
def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"\d+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()

    words = [word for word in words if word not in stop_words]

    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


# ---------------- Streamlit UI ---------------- #

st.set_page_config(
    page_title="AI Customer Support Log Analyser",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Customer Support Log Analyser")

st.markdown(
"""
This application uses **Machine Learning and NLP**
to automatically classify customer complaints into the appropriate product category.
"""
)

st.subheader("Enter Customer Complaint")

user_input = st.text_area(
    "",
    placeholder="Example: I was charged twice for my credit card payment and customer support has not responded.",
    height=180
)

if st.button("Predict Category"):

    if user_input.strip() == "":
        st.warning("Please enter a customer complaint.")
    else:

        clean_text = preprocess_text(user_input)

        vector = vectorizer.transform([clean_text])

        prediction = model.predict(vector)

        category = label_encoder.inverse_transform(prediction)

        st.success("Prediction Completed!")

        st.markdown("### Predicted Product Category")

        st.info(category[0])

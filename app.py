import streamlit as st

st.set_page_config(
    page_title="AI Customer Support Log Analyser",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# SIDEBAR
# ===============================

with st.sidebar:

    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)

    st.title("AI Support")

    st.markdown("---")

    st.markdown("## 📂 Navigation")

    st.page_link("app.py", label="🏠 Home", icon="🏠")
    st.page_link("pages/1_Dashboard.py", label="📊 Dashboard", icon="📊")
    st.page_link("pages/2_Data_Explorer.py", label="📁 Data Explorer", icon="📁")
    st.page_link("pages/3_AI_Predictor.py", label="🤖 AI Predictor", icon="🤖")
    st.page_link("pages/4_Model_Performance.py", label="📈 Model Performance", icon="📈")
    st.page_link("pages/5_About_Project.py", label="ℹ About", icon="ℹ️")

    st.markdown("---")

    st.success("🟢 Model Loaded")

    st.caption("Version 1.0")

import streamlit as st
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Research Abstract Classifier",
    page_icon="📚",
    layout="centered"
)

# Load the saved model and TF-IDF vectorizer
BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "best_research_classifier.pkl")
    vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

# Application interface
st.title("📚 Research Abstract Classifier")
st.write(
    "Classify a research abstract into Science, Business, "
    "or Social Science using machine learning."
)

abstract_text = st.text_area(
    "Paste your research abstract below:",
    height=220,
    placeholder="Enter or paste the abstract here..."
)

if st.button("Predict Category", type="primary"):
    if not abstract_text.strip():
        st.warning("Please enter an abstract first.")
    else:
        features = vectorizer.transform([abstract_text])
        prediction = model.predict(features)[0]

        st.success(f"Predicted Category: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]

            st.subheader("Model prediction scores")
            for category, probability in zip(
                model.classes_, probabilities
            ):
                st.write(f"**{category}:** {probability:.1%}")
                st.progress(float(probability))

st.divider()
st.caption(
    "Academic project — predictions depend on the training data "
    "and should be reviewed rather than treated as definitive."
)

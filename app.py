
import streamlit as st
import joblib
import pandas as pd
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
    model = joblib.load(
        BASE_DIR / "models" / "best_research_classifier.pkl"
    )
    vectorizer = joblib.load(
        BASE_DIR / "models" / "tfidf_vectorizer.pkl"
    )
    return model, vectorizer


model, vectorizer = load_artifacts()

st.title("📚 Research Abstract Classifier")
st.write(
    "Classify research abstracts into Science, Business, or Social Science "
    "using a trained machine-learning model."
)

tab_single, tab_csv = st.tabs(
    ["Single abstract", "Upload CSV dataset"]
)

# --------------------------------------------------
# TAB 1: SINGLE ABSTRACT PREDICTION
# --------------------------------------------------

with tab_single:
    st.subheader("Predict one research abstract")

    abstract_text = st.text_area(
        "Paste your research abstract below:",
        height=220,
        placeholder="Enter or paste the abstract here..."
    )

    if st.button(
        "Predict Category",
        type="primary",
        key="predict_single"
    ):
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


# --------------------------------------------------
# TAB 2: CSV BATCH PREDICTION
# --------------------------------------------------

with tab_csv:
    st.subheader("Upload a CSV file")

    st.write(
        "Upload a CSV containing a column of research abstracts. "
        "The app will predict a category for each non-empty abstract."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="abstract_csv_uploader"
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            if input_df.empty:
                st.warning("The uploaded CSV is empty.")

            else:
                st.write("Preview of uploaded data:")
                st.dataframe(input_df.head())

                # Identify columns containing text
                text_columns = [
                    column
                    for column in input_df.columns
                    if (
                        pd.api.types.is_string_dtype(input_df[column])
                        or input_df[column].dtype == "object"
                    )
                ]

                if not text_columns:
                    st.error(
                        "No text column was found. Please upload a CSV "
                        "with a column containing research abstracts."
                    )

                else:
                    # Automatically select the abstract column if available
                    normalized_columns = {
                        str(column).strip().lower(): column
                        for column in text_columns
                    }

                    preferred_names = [
                        "abstract",
                        "research_abstract",
                        "research abstract",
                        "text",
                        "content"
                    ]

                    default_column = next(
                        (
                            normalized_columns[name]
                            for name in preferred_names
                            if name in normalized_columns
                        ),
                        text_columns[0]
                    )

                    default_index = text_columns.index(default_column)

                    abstract_column = st.selectbox(
                        "Select the column containing research abstracts:",
                        options=text_columns,
                        index=default_index
                    )

                    if st.button(
                        "Predict categories for CSV",
                        type="primary",
                        key="predict_csv"
                    ):
                        result_df = input_df.copy()

                        abstract_series = (
                            result_df[abstract_column]
                            .fillna("")
                            .astype(str)
                            .str.strip()
                        )

                        valid_mask = abstract_series.ne("")

                        result_df["Predicted_Category"] = ""

                        if valid_mask.any():
                            features = vectorizer.transform(
                                abstract_series[valid_mask].tolist()
                            )

                            result_df.loc[
                                valid_mask, "Predicted_Category"
                            ] = model.predict(features)

                            if hasattr(model, "predict_proba"):
                                probabilities = model.predict_proba(
                                    features
                                )

                                for index, category in enumerate(
                                    model.classes_
                                ):
                                    result_df.loc[
                                        valid_mask,
                                        f"Score_{category.replace(' ', '_')}"
                                    ] = probabilities[:, index]

                        st.success(
                            f"Predictions completed for "
                            f"{int(valid_mask.sum())} non-empty abstracts."
                        )

                        st.dataframe(result_df.head(20))

                        # Download predictions
                        csv_bytes = result_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "Download predictions CSV",
                            data=csv_bytes,
                            file_name="research_abstract_predictions.csv",
                            mime="text/csv"
                        )

        except Exception as error:
            st.error(
                "Could not read or process this CSV. Please check that it is "
                f"a valid CSV file. Details: {error}"
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Academic project — predictions depend on the training data "
    "and should be reviewed rather than treated as definitive."
)


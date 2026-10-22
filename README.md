# Research Abstract Classifier

## Project Overview
A machine learning and Natural Language Processing (NLP) project that classifies research abstracts into three categories: Science, Business, and Social Science.

## Dataset
The dataset was collected using the OpenAlex API and filtered using keyword-based checks.

| Category | Number of Abstracts |
|---|---:|
| Science | 371 |
| Business | 352 |
| Social Science | 367 |
| **Total** | **1,090** |

The records have not all been independently verified manually as peer-reviewed research.

## Methodology
1. Collect research records using OpenAlex.
2. Clean and filter the abstracts.
3. Convert text into numerical features using TF-IDF.
4. Train and evaluate six machine learning models.
5. Select the best-performing model.
6. Build a web application using Streamlit.

## Machine Learning Models
- Logistic Regression
- Decision Tree
- Multinomial Naive Bayes
- K-Nearest Neighbors (KNN)
- Linear Support Vector Classifier (Linear SVC)
- Random Forest

## Results

The dataset was split into 80% training data and 20% testing data.

| Model | Test Accuracy |
|---|---:|
| Random Forest | 97.25% |
| Multinomial Naive Bayes | 96.33% |
| Linear SVC | 94.50% |
| Logistic Regression | 94.04% |
| K-Nearest Neighbors | 93.12% |
| Decision Tree | 91.74% |

Random Forest achieved the highest test accuracy.

**Limitation:** The high accuracy may partly reflect how the records were collected and labelled. Independent testing on manually labelled abstracts is needed to evaluate real-world performance.

## Web Application
The Streamlit application lets users paste a research abstract and view its predicted category and model prediction scores.

## Run Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Repository Files
- `app.py` — Streamlit application
- `requirements.txt` — Required Python packages
- `best_research_classifier.pkl` — Trained Random Forest model
- `tfidf_vectorizer.pkl` — Saved TF-IDF vectorizer

## Technologies Used
Python, Pandas, Scikit-learn, TF-IDF, Joblib, OpenAlex API, and Streamlit.

## Limitations
- The model may misclassify some abstracts.
- Some records may not be peer-reviewed research.
- Predictions should be reviewed rather than treated as definitive.

## Author
Academic project on machine learning-based research abstract classification.

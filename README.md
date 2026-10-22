# Research Abstract Classifier

A machine learning and Natural Language Processing (NLP) project that classifies research abstracts into **Science, Business, and Social Science** categories.

## 🚀 Live Demo

**[Open the Research Abstract Classifier](https://research-abstract-classifier-prince-chauhan.streamlit.app/)**

Try the application online by entering a research abstract to predict its category and view the model's prediction scores.

## 📌 Project Overview

This project uses machine learning and TF-IDF text features to classify research abstracts into three subject categories.

## 📊 Dataset

The dataset was collected using the OpenAlex API and filtered using keyword-based checks.

| Category | Number of Abstracts |
|---|---:|
| Science | 371 |
| Business | 352 |
| Social Science | 367 |
| **Total** | **1,090** |

**Data quality note:** The records have not all been independently verified manually as peer-reviewed research.

## ⚙️ Methodology

1. Collect research records using OpenAlex.
2. Clean and filter the abstracts.
3. Convert text into numerical features using TF-IDF.
4. Split the dataset into training and testing sets using an 80:20 split.
5. Train and evaluate six machine learning models.
6. Select the best-performing model.
7. Deploy the application using Streamlit.

## 🤖 Machine Learning Models

- Logistic Regression
- Decision Tree
- Multinomial Naive Bayes
- K-Nearest Neighbors (KNN)
- Linear Support Vector Classifier (Linear SVC)
- Random Forest

## 📈 Results

The following test accuracies were obtained during model evaluation.

| Model | Test Accuracy |
|---|---:|
| Random Forest | **97.25%** |
| Multinomial Naive Bayes | 96.33% |
| Linear SVC | 94.50% |
| Logistic Regression | 94.04% |
| K-Nearest Neighbors | 93.12% |
| Decision Tree | 91.74% |

Random Forest achieved the highest test accuracy among the evaluated models.

**Evaluation limitation:** The high accuracy may partly reflect how the records were collected and labelled. Testing on independently labelled abstracts is needed to better evaluate real-world performance.

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/PRINCE260902/research-abstract-classifier.git
cd research-abstract-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

## 📁 Repository Files

- `app.py` — Streamlit web application.
- `requirements.txt` — Python dependencies.
- `best_research_classifier.pkl` — Saved Random Forest model.
- `tfidf_vectorizer.pkl` — Saved TF-IDF vectorizer.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Joblib
- OpenAlex API
- Streamlit

## ⚠️ Limitations

- The model may misclassify some abstracts.
- Some records may not be peer-reviewed research.
- The high test accuracy may not reflect performance on all real-world abstracts.
- Predictions should be reviewed rather than treated as definitive.

## 👨‍💻 Author

Academic project on machine learning-based research abstract classification.

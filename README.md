
# Disease Prediction System (ML + Streamlit)

## Project Overview
This project is a machine learning-based disease prediction system that predicts diseases using patient symptoms and health-related information.

The project follows a complete machine learning pipeline including:

- Data Cleaning
- Feature Engineering
- SMOTE Balancing
- Feature Scaling
- Model Training
- Evaluation Metrics
- Cross Validation
- Streamlit Deployment

---

## Dataset
Kaggle Healthcare Disease Prediction Dataset

Dataset Link:
https://www.kaggle.com/datasets/algozee/healthcare-disease-prediction-dataset

---

## Models Used

### Required
- RandomForestClassifier

### Bonus Models
- Logistic Regression
- Support Vector Machine (SVM)

---

## Evaluation Metrics
The project evaluates the model using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Cross Validation
Stratified K-Fold Cross Validation is used to evaluate generalization performance.

---

## Project Structure

project/
│
├── train.py
├── app.py
├── model.pkl
├── label_encoder.pkl
├── scaler.pkl
├── feature_columns.pkl
├── requirements.txt
├── README.md
└── dataset.csv

---

## How to Run

### Install dependencies
pip install -r requirements.txt

### Train the model
python train.py

### Run Streamlit app
streamlit run app.py

---

## GUI Features
- User-friendly interface
- Predict disease from patient data
- Displays disease name clearly

---

## Bonus Features
- Feature Importance Visualization
- Confusion Matrix Heatmap
- Multiple Model Comparison

---

## Author
Machine Learning Assignment Project

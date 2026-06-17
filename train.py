
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE

# =========================
# 1. DATA LOADING
# =========================

df = pd.read_csv("dataset.csv")

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe(include='all'))

# =========================
# 2. DATA CLEANING
# =========================

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Fix inconsistent labels
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip().str.lower()

# =========================
# 3. FEATURE PROCESSING
# =========================

target_column = "disease"

X = df.drop(target_column, axis=1)
y = df[target_column]

# Encode categorical feature columns
X = pd.get_dummies(X)

# Label Encoding target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# =========================
# 4. TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# =========================
# 5. DATA BALANCING
# =========================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

# =========================
# 6. FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# =========================
# 7. MODEL TRAINING
# =========================

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train_smote)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train_smote)

# SVM
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train_smote)

# =========================
# 8. MODEL EVALUATION
# =========================

models = {
    "Random Forest": rf_model,
    "Logistic Regression": lr_model,
    "SVM": svm_model
}

for name, model in models.items():
    print(f"\n========== {name} ==========")

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"{name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()

# =========================
# 9. CROSS VALIDATION
# =========================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    rf_model,
    X,
    y_encoded,
    cv=skf,
    scoring='f1_weighted'
)

print("\nCross Validation F1 Scores:", cv_scores)
print("Mean F1 Score:", cv_scores.mean())

# =========================
# 10. FEATURE IMPORTANCE
# =========================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Feature Importance:")
print(feature_importance.head(10))

plt.figure(figsize=(12, 6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance.head(10)
)
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# =========================
# 11. SAVE MODEL
# =========================

joblib.dump(rf_model, "model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("\nModel and encoders saved successfully.")

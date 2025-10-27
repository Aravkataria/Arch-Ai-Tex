import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATASET_PATH = "C:/Users/ARAV/Desktop/idle/housing.csv"
MODEL_OUTPUT_FILENAME = "random_forest_classifier_model.joblib"

print(f"Loading dataset from {DATASET_PATH}...")
try:
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully.")
    print(f"Initial dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: {DATASET_PATH} not found.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

REQUIRED_COLUMNS = ['area', 'bedrooms']
for col in REQUIRED_COLUMNS:
    if col not in df.columns:
        print(f"Error: Column '{col}' not found. Available columns: {df.columns.tolist()}")
        exit()

df.dropna(subset=REQUIRED_COLUMNS, inplace=True)
df['bedrooms'] = df['bedrooms'].astype(int)

conditions = [
    (df['bedrooms'] <= 1) & (df['area'] < 1000),
    (df['bedrooms'] == 1) & (df['area'] >= 1000),
    (df['bedrooms'] == 2),
    (df['bedrooms'] == 3),
    (df['bedrooms'] >= 4) | (df['area'] >= 3500)
]
choices = [0, 1, 2, 3, 4]
df['dwelling_type'] = np.select(conditions, choices, default=4)

print(df['dwelling_type'].value_counts().sort_index())

X = df[['area', 'bedrooms']]
y = df['dwelling_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(rf_model, "random_forest_classifier_model.joblib")
print("Model saved as random_forest_classifier_model.joblib")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ----------------------------
# Config
# ----------------------------
DATASET_PATH = "housing.csv"  # CSV generated from pixel_area_mapping.py
MODEL_OUTPUT_FILENAME = "room_predictor.joblib"

# ----------------------------
# Load dataset
# ----------------------------
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

# ----------------------------
# Check required columns
# ----------------------------
REQUIRED_COLUMNS = ['pixel_count', 'relative_area']
for col in REQUIRED_COLUMNS:
    if col not in df.columns:
        print(f"Error: Column '{col}' not found. Available columns: {df.columns.tolist()}")
        exit()

df.dropna(subset=REQUIRED_COLUMNS, inplace=True)

# ----------------------------
# Features and target
# ----------------------------
X = df[['pixel_count']]          # Feature: pixel count of region
y = df['relative_area']          # Target: relative area (0-1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train Random Forest Regressor
# ----------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ----------------------------
# Predict and evaluate
# ----------------------------
y_pred = rf_model.predict(X_test)

print("Evaluation Metrics:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# ----------------------------
# Save model
# ----------------------------
joblib.dump(rf_model, MODEL_OUTPUT_FILENAME)
print(f"Model saved as {MODEL_OUTPUT_FILENAME}")

# ----------------------------
# Example: Predict new areas from pixel counts
# ----------------------------
sample_pixels = [5000, 12000, 2500]  # Example pixel counts
predicted_areas = rf_model.predict(pd.DataFrame({'pixel_count': sample_pixels}))
print("Predicted relative areas for sample pixels:", predicted_areas)

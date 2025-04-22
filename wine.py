import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# URLs for red and white wine datasets
url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

# Load data
df_red = pd.read_csv(url_red, sep=";")
df_white = pd.read_csv(url_white, sep=";")

# Add type column: 0 for red, 1 for white (optional, not used for prediction)
df_red["wine_type"] = 0
df_white["wine_type"] = 1

# Combine datasets
df = pd.concat([df_red, df_white], axis=0, ignore_index=True)

# Separate features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")


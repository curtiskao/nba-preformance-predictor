# train_model.py
"""
Trains a simple regression model to predict NBA player points per game
using engineered features from processed player data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# ----------------------------
# CONFIG
# ----------------------------
PLAYER_NAME = "Devin Booker"
INPUT_FILE = f"output/{PLAYER_NAME}_processed.csv"
MODEL_FILE = f"models/{PLAYER_NAME}_points_model.pkl"

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(INPUT_FILE)

# Drop rows with missing rolling averages (first 2 games)
df = df.dropna(subset=["PTS_last3", "REB_last3", "AST_last3"])

# ----------------------------
# FEATURES & TARGET
# ----------------------------
FEATURES = [
    "PTS_last3",
    "REB_last3",
    "AST_last3",
    "OPP_DEF_RATING",
    "OPP_PACE",
    "Is_Home",
    "Is_BackToBack",
]

TARGET = "PTS"

X = df[FEATURES]
y = df[TARGET]

# ----------------------------
# TRAIN / TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# TRAIN MODEL
# ----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    random_state=42
)
model.fit(X_train, y_train)

# ----------------------------
# EVALUATE
# ----------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Player: {PLAYER_NAME}")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f} points")

# ----------------------------
# SAVE MODEL
# ----------------------------
import os
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_FILE)
print(f"Trained model saved to {MODEL_FILE}")
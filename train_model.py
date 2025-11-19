# train_model.py
"""
Trains a regression model to predict NBA player points per game
using engineered features from processed player data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os


# ----------------------------
# CONFIG
# ----------------------------
PLAYER_NAME = "Devin Booker"
INPUT_FILE = f"output/{PLAYER_NAME}_processed.csv"
MODEL_FILE = f"models/{PLAYER_NAME}_points_model.pkl"

# Feature columns to use for prediction
FEATURES = [
    "PTS_last3",
    "REB_last3",
    "AST_last3",
    "FG_PCT_last3",
    "MIN_last3",
    "OPP_DEF_RATING",
    "OPP_PACE",
    "Is_Home",
    "Is_BackToBack",
]

TARGET = "PTS"


# ----------------------------
# LOAD DATA
# ----------------------------
print(f"Loading data for {PLAYER_NAME}...")
df = pd.read_csv(INPUT_FILE)

# Ensure date is datetime and sort chronologically
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values("GAME_DATE").reset_index(drop=True)

print(f"Loaded {len(df)} games")

# Drop rows with missing features
df = df.dropna(subset=FEATURES + [TARGET])
print(f"After removing NaN: {len(df)} games")


# ----------------------------
# TIME-BASED TRAIN/TEST SPLIT
# ----------------------------
# Use 80% earliest games for training, 20% most recent for testing
# This prevents data leakage and simulates real prediction scenario

train_size = int(0.8 * len(df))

train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print(f"\nTrain set: {len(X_train)} games (up to {train_df['GAME_DATE'].max().date()})")
print(f"Test set:  {len(X_test)} games (from {test_df['GAME_DATE'].min().date()})")


# ----------------------------
# TRAIN MODEL
# ----------------------------
print("\nTraining Random Forest model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

model.fit(X_train, y_train)
print("✓ Model trained")


# ----------------------------
# EVALUATE ON TEST SET
# ----------------------------
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Training set performance
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print(f"\nTraining Set:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  MAE:      {train_mae:.2f} points")
print(f"  RMSE:     {train_rmse:.2f} points")

# Test set performance
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\nTest Set:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  MAE:      {test_mae:.2f} points")
print(f"  RMSE:     {test_rmse:.2f} points")

# Check for overfitting
if train_r2 - test_r2 > 0.2:
    print("\n⚠️  Warning: Possible overfitting (train R² >> test R²)")


# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

feature_importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n" + feature_importance.to_string(index=False))


# ----------------------------
# SAMPLE PREDICTIONS
# ----------------------------
print("\n" + "="*60)
print("SAMPLE PREDICTIONS (Recent Test Games)")
print("="*60)

# Show last 5 test games
test_sample = test_df.tail(5).copy()
test_sample["Predicted_PTS"] = model.predict(test_sample[FEATURES])
test_sample["Error"] = test_sample["Predicted_PTS"] - test_sample[TARGET]

display_cols = ["GAME_DATE", "MATCHUP", TARGET, "Predicted_PTS", "Error", "Is_Home"]
print("\n" + test_sample[display_cols].to_string(index=False))


# ----------------------------
# SAVE MODEL
# ----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_FILE)

# Also save feature list for prediction script
feature_metadata = {
    "player_name": PLAYER_NAME,
    "features": FEATURES,
    "target": TARGET,
    "train_date_range": (
        str(train_df["GAME_DATE"].min().date()),
        str(train_df["GAME_DATE"].max().date())
    ),
    "test_date_range": (
        str(test_df["GAME_DATE"].min().date()),
        str(test_df["GAME_DATE"].max().date())
    ),
    "test_mae": float(test_mae),
    "test_r2": float(test_r2)
}

import json
metadata_file = MODEL_FILE.replace(".pkl", "_metadata.json")
with open(metadata_file, "w") as f:
    json.dump(feature_metadata, f, indent=2)

print(f"\n✓ Model saved to {MODEL_FILE}")
print(f"✓ Metadata saved to {metadata_file}")
print("\nDone!")

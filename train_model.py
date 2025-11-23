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
import json


# ----------------------------
# CONFIG
# ----------------------------
PLAYER_NAME = "Devin Booker"
INPUT_FILE = f"output/{PLAYER_NAME}_processed.csv"
MODEL_FILE = f"models/{PLAYER_NAME}_points_model.pkl"

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


# ----------------------------
# CHECK MINIMUM DATA REQUIREMENT
# ----------------------------
MIN_GAMES = 8  # Need at least 8 games for 80/20 split

if len(df) < MIN_GAMES:
    print(f"\n❌ ERROR: Not enough data to train!")
    print(f"   You have {len(df)} games, but need at least {MIN_GAMES}.")
    print(f"\n   Options:")
    print(f"   1. Wait for more games to be played this season")
    print(f"   2. Fetch data from previous seasons")
    print(f"   3. Use a different player with more games")
    exit(1)


# ----------------------------
# DEFINE FEATURES
# ----------------------------
ALL_FEATURES = [
    # Rolling averages
    "PTS_last3", "MIN_last3", "FGA_last3",
    "REB_last3", "AST_last3", "FG_PCT_last3",
    
    # Opponent strength
    "OPP_DEF_RATING", "OPP_OFF_RATING", "OPP_NET_RATING", "OPP_PACE",
    
    # Schedule
    "Is_Home", "Is_BackToBack", "Days_Rest",
    
    # Time
    "Game_Number", "Month", "Day_of_Week",
]

# Only use features that actually exist
FEATURES = [f for f in ALL_FEATURES if f in df.columns]
missing_features = [f for f in ALL_FEATURES if f not in df.columns]

print(f"\nUsing {len(FEATURES)} features")
if missing_features:
    print(f"⚠️  Missing features (will be skipped): {missing_features}")


# ----------------------------
# PREPARE DATA
# ----------------------------
# Drop rows with missing target
df = df.dropna(subset=[TARGET])
print(f"After removing NaN targets: {len(df)} games")


# ----------------------------
# TIME-BASED TRAIN/TEST SPLIT
# ----------------------------
train_size = int(0.7 * len(df))  # Use 70/30 split for small datasets

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

# Use simpler model for small datasets
model = RandomForestRegressor(
    n_estimators=100,  # Reduced from 200
    max_depth=5,       # Reduced from 10 to prevent overfitting
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✓ Model trained")


# ----------------------------
# EVALUATE
# ----------------------------
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Training set
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print(f"\nTraining Set:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  MAE:      {train_mae:.2f} points")
print(f"  RMSE:     {train_rmse:.2f} points")

# Test set
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\nTest Set:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  MAE:      {test_mae:.2f} points")
print(f"  RMSE:     {test_rmse:.2f} points")


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
print("SAMPLE PREDICTIONS")
print("="*60)

test_sample = test_df.copy()
test_sample["Predicted_PTS"] = model.predict(test_sample[FEATURES])
test_sample["Error"] = test_sample["Predicted_PTS"] - test_sample[TARGET]

display_cols = ["GAME_DATE", "MATCHUP", TARGET, "Predicted_PTS", "Error"]
print("\n" + test_sample[display_cols].to_string(index=False))


# ----------------------------
# SAVE MODEL
# ----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_FILE)

feature_metadata = {
    "player_name": PLAYER_NAME,
    "features": FEATURES,
    "target": TARGET,
    "train_games": len(train_df),
    "test_games": len(test_df),
    "test_r2": float(test_r2),
    "test_mae": float(test_mae),
}

metadata_file = MODEL_FILE.replace(".pkl", "_metadata.json")
with open(metadata_file, "w") as f:
    json.dump(feature_metadata, f, indent=2)

print("\n" + "="*60)
print(f"✓ Model saved to {MODEL_FILE}")
print(f"✓ Metadata saved to {metadata_file}")
print("\nDone!")
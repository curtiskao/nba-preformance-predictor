# train_model.py
"""
Trains an XGBoost model to predict NBA player points.
XGBoost is used as the default algorithm for better performance.
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import json

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")
    print("   Falling back to Random Forest...")
    from sklearn.ensemble import RandomForestRegressor


def train_model(player_name):
    """Train XGBoost model for player points prediction."""
    
    # ----------------------------
    # CONFIG
    # ----------------------------
    INPUT_FILE = f"output/{player_name}_processed.csv"
    MODEL_FILE = f"models/{player_name}_model.pkl"
    SCALER_FILE = f"models/{player_name}_scaler.pkl"
    TARGET = "PTS"

    # ----------------------------
    # LOAD DATA
    # ----------------------------
    print(f"Loading data for {player_name}...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ ERROR: Processed data not found at {INPUT_FILE}")
        print(f"Run this first: python process_data.py --player \"{player_name}\"")
        exit(1)
    
    df = pd.read_csv(INPUT_FILE)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    print(f"Loaded {len(df)} games")
    print(f"Date range: {df['GAME_DATE'].min().date()} to {df['GAME_DATE'].max().date()}")

    # ----------------------------
    # CHECK MINIMUM DATA
    # ----------------------------
    MIN_GAMES = 50  # Increased from 30 for 5 seasons

    if len(df) < MIN_GAMES:
        print(f"\n⚠️  WARNING: Only {len(df)} games. Recommended: {MIN_GAMES}+")
        if len(df) < 30:
            print(f"\n❌ ERROR: Need at least 30 games. You have {len(df)}.")
            exit(1)

    # ----------------------------
    # DEFINE FEATURES
    # ----------------------------
    # Use all available L3, L5, L10, OPP, and context features
    FEATURE_KEYWORDS = ['_L3', '_L5', '_L10', 'OPP_', 'Is_', 'TREND', 'STD', 
                       'PER_MIN', 'USAGE', 'TS_', 'FG3', 'Days_Rest']
    
    # Get potential features
    potential_features = [col for col in df.columns if any(kw in col for kw in FEATURE_KEYWORDS)]
    
    # Filter to only numeric columns (exclude OPP_TEAM_NAME, etc.)
    FEATURES = [col for col in potential_features if pd.api.types.is_numeric_dtype(df[col])]
    
    print(f"\nUsing {len(FEATURES)} features")
    if len(FEATURES) <= 15:
        for feat in FEATURES:
            print(f"  - {feat}")
    else:
        print(f"  (Showing first 15)")
        for feat in FEATURES[:15]:
            print(f"  - {feat}")
        print(f"  ... and {len(FEATURES) - 15} more")

    # ----------------------------
    # PREPARE DATA
    # ----------------------------
    df = df.dropna(subset=[TARGET])
    
    # Clean data - only for numeric features
    print(f"\nCleaning data...")
    for feat in FEATURES:
        if feat in df.columns:
            # Replace inf with NaN
            df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)
            # Fill NaN with median
            if df[feat].notna().sum() > 0:  # Only if there are non-NaN values
                df[feat] = df[feat].fillna(df[feat].median())
            else:
                df[feat] = df[feat].fillna(0)
    
    # Drop rows with remaining NaN values in features
    initial_len = len(df)
    df = df.dropna(subset=FEATURES)
    if len(df) < initial_len:
        print(f"  Removed {initial_len - len(df)} rows with missing values")
    
    print(f"After cleaning: {len(df)} games")

    # ----------------------------
    # TRAIN/TEST SPLIT
    # ----------------------------
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
    # SCALE FEATURES
    # ----------------------------
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------------------------
    # TRAIN MODEL
    # ----------------------------
    print("\n" + "="*60)
    if XGBOOST_AVAILABLE:
        print("TRAINING XGBOOST MODEL")
    else:
        print("TRAINING RANDOM FOREST MODEL")
    print("="*60 + "\n")
    
    if XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(
            n_estimators=75,         # Increased from 50 (more data = can handle more trees)
            max_depth=4,             # Increased from 3 (more data = can handle slightly more complexity)
            learning_rate=0.05,      # Keep slow learning rate
            subsample=0.7,           # Keep randomness
            colsample_bytree=0.7,    # Keep feature sampling
            min_child_weight=3,      # Keep minimum samples per leaf
            gamma=0.1,               # Keep pruning
            reg_alpha=0.3,           # Reduced from 0.5 (more data = need less regularization)
            reg_lambda=0.8,          # Reduced from 1.0 (more data = need less regularization)
            random_state=42,
            objective='reg:squarederror',
            early_stopping_rounds=15  # Increased from 10 (more patience with more data)
        )
        
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Get number of trees used
        if hasattr(model, 'best_iteration'):
            print(f"✓ Model trained (used {model.best_iteration} trees)")
        else:
            print(f"✓ Model trained")
        
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
    
    print("✓ Model trained")

    # ----------------------------
    # EVALUATE
    # ----------------------------
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)

    # Training set
    y_train_pred = model.predict(X_train_scaled)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    print(f"\nTraining Set ({len(y_train)} games):")
    print(f"  R² Score: {train_r2:.4f}")
    print(f"  MAE:      {train_mae:.2f} points")
    print(f"  RMSE:     {train_rmse:.2f} points")

    # Test set
    y_test_pred = model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"\nTest Set ({len(y_test)} games):")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  MAE:      {test_mae:.2f} points")
    print(f"  RMSE:     {test_rmse:.2f} points")
    
    # Interpretation
    if test_r2 < 0:
        print("\n⚠️  WARNING: Negative R²!")
        print("   Model worse than baseline. Try more data or simpler model.")
    elif test_r2 > 0.5:
        print(f"\n✓ Excellent! Model explains {test_r2*100:.1f}% of variance")
    elif test_r2 > 0.3:
        print(f"\n✓ Good! Model explains {test_r2*100:.1f}% of variance")
    else:
        print(f"\n⚠️  Fair. Model explains {test_r2*100:.1f}% of variance")
    
    # Baseline comparison
    baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    
    print(f"\nBaseline (always predict {y_train.mean():.1f}):")
    print(f"  MAE: {baseline_mae:.2f} points")
    
    if test_mae < baseline_mae:
        improvement = ((baseline_mae - test_mae) / baseline_mae) * 100
        print(f"  ✓ Model is {improvement:.1f}% better than baseline!")
    
    # Reference model comparison
    reference_mae = 4.77
    print(f"\nReference Model (100k games, XGBoost): MAE = {reference_mae:.2f}")
    if test_mae < reference_mae + 0.5:
        print(f"  🎉 You're competitive with professional models!")
    elif test_mae < reference_mae + 1.5:
        print(f"  ✓ Good performance for single-player model!")

    # ----------------------------
    # FEATURE IMPORTANCE
    # ----------------------------
    print("\n" + "="*60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*60)

    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        print("\n" + feature_importance.head(10).to_string(index=False))
        
        # Save full importance
        feature_importance.to_csv(
            MODEL_FILE.replace(".pkl", "_feature_importance.csv"),
            index=False
        )

    # ----------------------------
    # SAMPLE PREDICTIONS
    # ----------------------------
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS (Last 10 test games)")
    print("="*60)

    test_sample = test_df.tail(10).copy()
    test_sample["Predicted_PTS"] = model.predict(scaler.transform(test_sample[FEATURES]))
    test_sample["Error"] = test_sample["Predicted_PTS"] - test_sample[TARGET]

    display_cols = ["GAME_DATE", "MATCHUP", TARGET, "Predicted_PTS", "Error"]
    print("\n" + test_sample[display_cols].to_string(index=False))

    # ----------------------------
    # SAVE MODEL
    # ----------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    metadata = {
        "player_name": player_name,
        "features": FEATURES,
        "model_type": "XGBoost" if XGBOOST_AVAILABLE else "RandomForest",
        "train_games": len(train_df),
        "test_games": len(test_df),
        "test_r2": float(test_r2),
        "test_mae": float(test_mae),
        "date_range": {
            "start": str(df["GAME_DATE"].min().date()),
            "end": str(df["GAME_DATE"].max().date())
        }
    }

    metadata_file = MODEL_FILE.replace(".pkl", "_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*60)
    print(f"✓ Model saved to {MODEL_FILE}")
    print(f"✓ Scaler saved to {SCALER_FILE}")
    print(f"✓ Metadata saved to {metadata_file}")
    print("="*60)
    
    print(f"\nNext: python prediction.py --player \"{player_name}\" --opponent \"Team Name\" --home")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost model for NBA player points prediction"
    )
    parser.add_argument(
        "--player",
        type=str,
        default="Devin Booker",
        help="Player's full name"
    )
    
    args = parser.parse_args()
    
    if not XGBOOST_AVAILABLE:
        print("\n💡 TIP: Install XGBoost for better performance:")
        print("   pip install xgboost\n")
    
    train_model(args.player)
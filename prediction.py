# prediction.py
"""
Predicts next game points using trained XGBoost model.
"""

import pandas as pd
import numpy as np
import joblib
import json
import argparse
import os


def load_model(player_name):
    """Load trained model, scaler, and metadata."""
    model_path = f"models/{player_name}_model.pkl"
    scaler_path = f"models/{player_name}_scaler.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print(f"Train a model first: python train_model.py --player \"{player_name}\"")
        exit(1)

    print(f"✓ Loading model: {model_path}")
    model = joblib.load(model_path)
    
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"✓ Loading scaler: {scaler_path}")

    metadata_file = model_path.replace(".pkl", "_metadata.json")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    return model, scaler, metadata


def load_processed_data(player_name):
    """Load processed player data."""
    csv_path = f"output/{player_name}_processed.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Processed data not found: {csv_path}")
        print(f"Run this first: python process_data.py --player \"{player_name}\"")
        exit(1)

    df = pd.read_csv(csv_path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    
    return df


def load_team_stats():
    """Load team statistics."""
    team_stats_path = "output/team_stats.csv"
    if not os.path.exists(team_stats_path):
        print(f"❌ Team stats not found: {team_stats_path}")
        print("Run: python process_data.py --player \"Player Name\"")
        exit(1)
    
    df = pd.read_csv(team_stats_path)
    return df


def build_next_game_features(
    recent_games_df,
    opponent_team_name,
    team_stats_df,
    is_home=True,
    days_rest=2,
    feature_list=None
):
    """Build feature vector for next game."""
    
    feature_dict = {}
    
    # Get opponent stats
    opp_row = team_stats_df[team_stats_df["TEAM_NAME"] == opponent_team_name]
    if opp_row.empty:
        print(f"⚠️  Warning: '{opponent_team_name}' not found. Using defaults.")
        opp_def_rating = 110.0
        opp_off_rating = 110.0
        opp_net_rating = 0.0
        opp_pace = 100.0
    else:
        opp_row = opp_row.iloc[0]
        opp_def_rating = opp_row.get("DEF_RATING", 110.0)
        opp_off_rating = opp_row.get("OFF_RATING", 110.0)
        opp_net_rating = opp_row.get("NET_RATING", 0.0)
        opp_pace = opp_row.get("PACE", 100.0)
    
    # Get recent games for rolling averages
    last_3 = recent_games_df.tail(3)
    last_5 = recent_games_df.tail(5)
    last_10 = recent_games_df.tail(10)
    
    # Build features
    if feature_list:
        for feat in feature_list:
            # L3 features
            if "_L3" in feat:
                base = feat.replace("_L3", "")
                if base in last_3.columns:
                    feature_dict[feat] = last_3[base].mean()
                else:
                    feature_dict[feat] = 0
            
            # L5 features
            elif "_L5" in feat:
                base = feat.replace("_L5", "")
                if base in last_5.columns:
                    feature_dict[feat] = last_5[base].mean()
                else:
                    feature_dict[feat] = 0
            
            # L10 features
            elif "_L10" in feat:
                base = feat.replace("_L10", "")
                if base in last_10.columns:
                    feature_dict[feat] = last_10[base].mean()
                else:
                    feature_dict[feat] = 0
            
            # Opponent features
            elif feat == "OPP_DEF_RATING":
                feature_dict[feat] = opp_def_rating
            elif feat == "OPP_OFF_RATING":
                feature_dict[feat] = opp_off_rating
            elif feat == "OPP_NET_RATING":
                feature_dict[feat] = opp_net_rating
            elif feat == "OPP_PACE":
                feature_dict[feat] = opp_pace
            
            # Game context
            elif feat == "Is_Home":
                feature_dict[feat] = 1 if is_home else 0
            elif feat == "Is_BackToBack":
                feature_dict[feat] = 1 if days_rest == 1 else 0
            elif feat == "Days_Rest":
                feature_dict[feat] = days_rest
            
            # Trend features
            elif feat == "PTS_TREND":
                pts_l3 = feature_dict.get("PTS_L3", 0)
                pts_l10 = feature_dict.get("PTS_L10", pts_l3)
                feature_dict[feat] = pts_l3 - pts_l10
            elif feat == "MIN_TREND":
                min_l3 = feature_dict.get("MIN_L3", 0)
                min_l10 = feature_dict.get("MIN_L10", min_l3)
                feature_dict[feat] = min_l3 - min_l10
            
            # Standard deviation
            elif "STD" in feat:
                base = feat.replace("_STD_L10", "")
                if base in last_10.columns:
                    feature_dict[feat] = last_10[base].std()
                else:
                    feature_dict[feat] = 0
            
            # Efficiency features
            elif feat == "PTS_PER_MIN_L5":
                pts = feature_dict.get("PTS_L5", 0)
                mins = feature_dict.get("MIN_L5", 1)
                feature_dict[feat] = pts / max(mins, 1)
            
            # Default
            else:
                feature_dict[feat] = 0
    
    # Clean inf/nan values
    for key in feature_dict:
        if np.isnan(feature_dict[key]) or np.isinf(feature_dict[key]):
            feature_dict[key] = 0
    
    return feature_dict


def predict_next_game(
    player_name,
    opponent_team_name,
    is_home=True,
    days_rest=2
):
    """Make prediction for next game."""
    
    # Load model
    model, scaler, metadata = load_model(player_name)
    features = metadata["features"]

    print(f"\n{'='*60}")
    print(f"PREDICTING NEXT GAME: {player_name}")
    print(f"{'='*60}\n")

    # Load data
    df = load_processed_data(player_name)
    print(f"✓ Loaded {len(df)} games")
    print(f"  Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
    
    last_game = df.iloc[-1]
    print(f"✓ Last game: {last_game['GAME_DATE'].date()} - {last_game['MATCHUP']} ({last_game['PTS']} pts)")

    team_stats = load_team_stats()
    print(f"✓ Loaded stats for {len(team_stats)} teams\n")

    # Build features
    print(f"Next game details:")
    print(f"  Opponent: {opponent_team_name}")
    print(f"  Location: {'Home' if is_home else 'Away'}")
    print(f"  Days rest: {days_rest}\n")
    
    feature_dict = build_next_game_features(
        recent_games_df=df,
        opponent_team_name=opponent_team_name,
        team_stats_df=team_stats,
        is_home=is_home,
        days_rest=days_rest,
        feature_list=features
    )

    # Convert to DataFrame
    X = pd.DataFrame([feature_dict], columns=features)
    
    # Scale
    if scaler is not None:
        X = scaler.transform(X)

    # Predict
    predicted_points = model.predict(X)[0]      

    # Display results
    print(f"{'='*60}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*60}\n")
    
    print(f"🎯 Predicted Points: {predicted_points:.1f}\n")
    
    print(f"Recent Performance (last 3-5 games):")
    print(f"  PTS (L3): {feature_dict.get('PTS_L3', 0):.1f}")
    print(f"  PTS (L5): {feature_dict.get('PTS_L5', 0):.1f}")
    print(f"  MIN (L5): {feature_dict.get('MIN_L5', 0):.1f}")
    print(f"  FGA (L5): {feature_dict.get('FGA_L5', 0):.1f}")
    
    print(f"\nOpponent ({opponent_team_name}):")
    print(f"  DEF Rating: {feature_dict.get('OPP_DEF_RATING', 0):.1f}")
    print(f"  PACE: {feature_dict.get('OPP_PACE', 0):.1f}")
    
    print(f"\nModel Info:")
    print(f"  Algorithm: {metadata['model_type']}")
    print(f"  Training games: {metadata['train_games']}")
    print(f"  Test MAE: {metadata['test_mae']:.2f} points")
    print(f"  Test R²: {metadata['test_r2']:.4f}")
    
    print(f"\n{'='*60}")
    print("✓ Prediction complete!")
    print(f"{'='*60}\n")
    
    return predicted_points, feature_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict NBA player points for next game"
    )
    parser.add_argument(
        "--player",
        type=str,
        default="Devin Booker",
        help="Player's full name"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        required=True,
        help="Full opponent team name (e.g., 'Los Angeles Lakers')"
    )
    parser.add_argument(
        "--home",
        action="store_true",
        help="Is this a home game?"
    )
    parser.add_argument(
        "--rest",
        type=int,
        default=2,
        help="Days of rest since last game (default: 2)"
    )
    
    args = parser.parse_args()

    predict_next_game(
        player_name=args.player,
        opponent_team_name=args.opponent,
        is_home=args.home,
        days_rest=args.rest
    )
"""
prediction.py
-------------
Predicts the next game's points for a player based on:
1. Their recent performance (rolling averages)
2. The upcoming opponent's stats
3. Game context (home/away, rest days)
"""

import pandas as pd
import numpy as np
import joblib
import json
import argparse
import os


# -------------------------------------------------------
# LOAD MODEL + METADATA
# -------------------------------------------------------
def load_model(player_name):
    model_path = f"models/{player_name}_points_model.pkl"
    scaler_path = f"models/{player_name}_scaler.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print(f"Train a model first: python train_model.py --player \"{player_name}\"")
        exit(1)

    print(f"✓ Loading model: {model_path}")
    model = joblib.load(model_path)
    
    # Load scaler if it exists
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"✓ Loading scaler: {scaler_path}")

    metadata_file = model_path.replace(".pkl", "_metadata.json")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    return model, scaler, metadata


# -------------------------------------------------------
# LOAD PROCESSED PLAYER DATA
# -------------------------------------------------------
def load_processed_data(player_name):
    csv_path = f"output/{player_name}_processed.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Processed data not found: {csv_path}")
        print(f"Run this first: python process_data.py --player \"{player_name}\"")
        exit(1)

    df = pd.read_csv(csv_path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    
    return df


# -------------------------------------------------------
# LOAD TEAM STATS
# -------------------------------------------------------
def load_team_stats():
    """Load pre-fetched team stats from output folder."""
    team_stats_path = "output/team_stats.csv"
    if not os.path.exists(team_stats_path):
        print(f"❌ Team stats not found: {team_stats_path}")
        print("Run this first: python process_data.py --player \"Player Name\"")
        exit(1)
    
    df = pd.read_csv(team_stats_path)
    return df


# -------------------------------------------------------
# BUILD FEATURE VECTOR FOR NEXT GAME
# -------------------------------------------------------
def build_next_game_features(
    recent_games_df,
    opponent_team_name,
    team_stats_df,
    is_home=True,
    days_rest=2,
    feature_list=None
):
    """
    Constructs feature vector for the next game.
    """
    
    feature_dict = {}
    
    # Get opponent stats
    opp_row = team_stats_df[team_stats_df["TEAM_NAME"] == opponent_team_name]
    if opp_row.empty:
        print(f"⚠️  Warning: Opponent '{opponent_team_name}' not found in team stats.")
        print("    Using default values for opponent features.")
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
    
    # Calculate rolling averages from recent games
    last_3_games = recent_games_df.tail(3)
    
    # Build features
    if feature_list:
        for feat in feature_list:
            # Rolling averages
            if feat == "PTS_last3":
                feature_dict[feat] = last_3_games["PTS"].mean()
            elif feat == "MIN_last3":
                feature_dict[feat] = last_3_games["MIN"].mean()
            elif feat == "FGA_last3":
                feature_dict[feat] = last_3_games["FGA"].mean()
            elif feat == "REB_last3":
                feature_dict[feat] = last_3_games["REB"].mean()
            elif feat == "AST_last3":
                feature_dict[feat] = last_3_games["AST"].mean()
            elif feat == "FG_PCT_last3":
                feature_dict[feat] = last_3_games["FG_PCT"].mean()
            
            # Opponent features
            elif feat == "OPP_DEF_RATING":
                feature_dict[feat] = opp_def_rating
            elif feat == "OPP_OFF_RATING":
                feature_dict[feat] = opp_off_rating
            elif feat == "OPP_NET_RATING":
                feature_dict[feat] = opp_net_rating
            elif feat == "OPP_PACE":
                feature_dict[feat] = opp_pace
            
            # Schedule features
            elif feat == "Is_Home":
                feature_dict[feat] = 1 if is_home else 0
            elif feat == "Is_BackToBack":
                feature_dict[feat] = 1 if days_rest == 1 else 0
            elif feat == "Days_Rest":
                feature_dict[feat] = days_rest
            
            else:
                feature_dict[feat] = 0.0
                print(f"⚠️  Unknown feature '{feat}', using 0")
    
    return feature_dict


# -------------------------------------------------------
# MAIN PREDICTION LOGIC
# -------------------------------------------------------
def predict_next_game(
    player_name,
    opponent_team_name,
    is_home=True,
    days_rest=2
):
    """
    Predict points for next game.
    """
    
    # 1. Load model + metadata + scaler
    model, scaler, metadata = load_model(player_name)  # Changed this line
    features = metadata["features"]

    print(f"\n{'='*60}")
    print(f"PREDICTING NEXT GAME FOR: {player_name}")
    print(f"{'='*60}\n")

    # 2. Load player's recent games
    df = load_processed_data(player_name)
    print(f"✓ Loaded {len(df)} recent games")
    
    last_game = df.iloc[-1]
    print(f"✓ Last game: {last_game['GAME_DATE'].date()} - {last_game['MATCHUP']} ({last_game['PTS']} pts)")

    # 3. Load team stats
    team_stats = load_team_stats()
    print(f"✓ Loaded stats for {len(team_stats)} teams\n")

    # 4. Build next game features
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

    # 5. Convert to DataFrame (preserves feature names, fixes warning)
    X = pd.DataFrame([feature_dict], columns=features)
    
    # Scale if scaler exists
    if scaler is not None:
        X = scaler.transform(X)

    # 6. Make prediction
    predicted_points = model.predict(X)[0]      

    # 7. Display results
    print(f"{'='*60}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*60}\n")
    
    print(f"🎯 Predicted Points: {predicted_points:.1f}\n")
    
    print(f"Recent Performance (last 3 games avg):")
    print(f"  PTS: {feature_dict.get('PTS_last3', 0):.1f}")
    print(f"  MIN: {feature_dict.get('MIN_last3', 0):.1f}")
    print(f"  FGA: {feature_dict.get('FGA_last3', 0):.1f}")
    
    print(f"\nOpponent Strength ({opponent_team_name}):")
    print(f"  DEF Rating: {feature_dict.get('OPP_DEF_RATING', 0):.1f}")
    print(f"  PACE: {feature_dict.get('OPP_PACE', 0):.1f}")
    
    print(f"\nAll features used:")
    for feat, val in feature_dict.items():
        print(f"  {feat:20s} = {val:.2f}")
    
    print(f"\n{'='*60}")
    print("✓ Prediction complete!")
    print(f"{'='*60}\n")
    
    return predicted_points, feature_dict


# -------------------------------------------------------
# CLI Support
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict NBA player points for next game"
    )
    parser.add_argument(
        "--player",
        type=str,
        default="Devin Booker",
        help="Player's full name (e.g., 'LeBron James', 'Stephen Curry')"
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
        help="Is this a home game? (default: away)"
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
# simple_predict.py
import pandas as pd
import argparse

def simple_prediction(player_name, opponent_name, is_home, days_rest):
    """Simple but effective prediction using recent averages."""
    
    # Load data
    df = pd.read_csv(f"output/{player_name}_processed.csv")
    df = df.sort_values("GAME_DATE")
    
    team_stats = pd.read_csv("output/team_stats.csv")
    
    # Base prediction: recent 5-game average
    recent_pts = df.tail(5)["PTS"].mean()
    
    # Opponent adjustment
    opp_stats = team_stats[team_stats["TEAM_NAME"] == opponent_name]
    if not opp_stats.empty:
        opp_def = opp_stats.iloc[0]["DEF_RATING"]
        league_avg_def = 110.0
        # Worse defense = more points
        def_adjustment = (league_avg_def - opp_def) * 0.15
    else:
        def_adjustment = 0
    
    # Home court advantage
    home_bonus = 2.5 if is_home else 0
    
    # Back-to-back penalty
    fatigue_penalty = -3.0 if days_rest == 1 else 0
    
    # Final prediction
    prediction = recent_pts + def_adjustment + home_bonus + fatigue_penalty
    
    print(f"\n{'='*60}")
    print(f"PREDICTION FOR: {player_name}")
    print(f"{'='*60}\n")
    print(f"Recent average (5 games):     {recent_pts:.1f} pts")
    print(f"Opponent defense adjustment:  {def_adjustment:+.1f} pts")
    print(f"Home court advantage:         {home_bonus:+.1f} pts")
    print(f"Fatigue penalty:              {fatigue_penalty:+.1f} pts")
    print(f"\n🎯 PREDICTED POINTS: {prediction:.1f}\n")
    print(f"{'='*60}\n")
    
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--home", action="store_true")
    parser.add_argument("--rest", type=int, default=2)
    args = parser.parse_args()
    
    simple_prediction(args.player, args.opponent, args.home, args.rest)
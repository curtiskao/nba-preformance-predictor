NBA Performance Predictor
A machine learning project that predicts NBA player point totals for upcoming games based on recent performance, opponent strength, and game context.
Overview
This system uses Ridge Regression to predict player scoring based on:

Recent Performance: Rolling averages (points, minutes, field goal attempts, rebounds, assists, FG%)
Opponent Strength: Defensive rating, offensive rating, net rating, and pace
Game Context: Home/away games, days of rest, back-to-back games

# install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# to remove old data
rm output/*_processed.csv
rm models/*

# run 
# For Devin Booker (default)
python process_data.py
python train_model.py
python prediction.py --opponent "Los Angeles Lakers" --home --rest 2

# For LeBron James
python process_data.py --player "LeBron James"
python train_model.py --player "LeBron James"
python prediction.py --player "LeBron James" --opponent "Boston Celtics" --rest 1
# nba-preformance-predictor
Uses machine learning to predict nba players' performances

# install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run 
python get_stats.py
python process_data.py
python train_model.py
# Predict home game vs Lakers with 2 days rest
python predict_points.py --opponent "Los Angeles Lakers" --home --rest 2

# Predict away game vs Warriors on back-to-back
python predict_points.py --opponent "Golden State Warriors" --rest 1

# Predict with 3 days rest (well-rested)
python predict_points.py --opponent "Boston Celtics" --home --rest 3
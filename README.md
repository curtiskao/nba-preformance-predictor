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
python predict_points.py --model models/Devin\ Booker_points_model.pkl 

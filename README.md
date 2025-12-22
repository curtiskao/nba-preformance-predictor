# nba-preformance-predictor
Uses machine learning to predict nba players' performances

# install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run 
# For Devin Booker (default)
python process_data.py
python train_model.py
python prediction.py --opponent "Los Angeles Lakers" --home --rest 2

# For LeBron James
python process_data.py --player "LeBron James"
python train_model.py --player "LeBron James"
python prediction.py --player "LeBron James" --opponent "Boston Celtics" --rest 1
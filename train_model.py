import json
import pickle

import pandas as pd
from xgboost import XGBRegressor

CONSISTENT_TEAMS = [
    'Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
    'Delhi Daredevils', 'Sunrisers Hyderabad'
]

df = pd.read_csv('ipl.csv')

df = df[(df['bat_team'].isin(CONSISTENT_TEAMS)) & (df['bowl_team'].isin(CONSISTENT_TEAMS))]
df = df[df['overs'] >= 5.0]

df = df.drop(['mid', 'date', 'venue', 'batsman', 'bowler'], axis=1)
df = pd.get_dummies(df, columns=['bat_team', 'bowl_team'])

X = df.drop('total', axis=1)
y = df['total']

model = XGBRegressor(random_state=42, n_estimators=200, max_depth=8, learning_rate=0.1)
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model_columns.json', 'w') as f:
    json.dump(X.columns.tolist(), f)

print(f"Trained on {X.shape[0]} rows, {X.shape[1]} features")
print("Saved model.pkl and model_columns.json")

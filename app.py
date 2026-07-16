import json
import pickle
import os

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'model_columns.json')) as f:
    model_columns = json.load(f)

TEAMS = [
    'Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab',
    'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        bat_team = request.form['bat_team']
        bowl_team = request.form['bowl_team']

        if bat_team == bowl_team:
            error = "Batting and bowling team can't be the same"
        else:
            try:
                overs = float(request.form['overs'])
                runs = int(request.form['runs'])
                wickets = int(request.form['wickets'])
                runs_last_5 = int(request.form['runs_last_5'])
                wickets_last_5 = int(request.form['wickets_last_5'])
                striker = int(request.form['striker'])
                non_striker = int(request.form['non_striker'])

                if overs < 5.0:
                    error = 'Prediction only works from the 5th over onward'
                else:
                    row = {col: 0 for col in model_columns}
                    row['runs'] = runs
                    row['wickets'] = wickets
                    row['overs'] = overs
                    row['runs_last_5'] = runs_last_5
                    row['wickets_last_5'] = wickets_last_5
                    row['striker'] = striker
                    row['non-striker'] = non_striker
                    row[f'bat_team_{bat_team}'] = 1
                    row[f'bowl_team_{bowl_team}'] = 1

                    X = pd.DataFrame([row])[model_columns]
                    prediction = int(model.predict(X)[0])
            except ValueError:
                error = 'Please enter valid numbers for all fields'

    return render_template('index.html', teams=TEAMS, prediction=prediction, error=error)


if __name__ == '__main__':
    app.run(debug=True)

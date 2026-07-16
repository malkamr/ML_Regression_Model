# ML Regression Model


# IPL First Innings Score Predictor

Predicts the final first-innings score of an IPL match from the current match state (overs bowled, runs, wickets, run rate in the last 5 overs, and the two teams). Built with XGBoost and served through a small Flask app.

## How it works

The model is trained on ball-by-ball IPL data (2008–2017). For each ball, the current state of the innings — runs, wickets, overs completed, runs/wickets in the last 5 overs, batting and bowling team — is used to predict what the final total will be.

Only the first 8 franchises with a consistent multi-season history are kept (teams like Kochi Tuskers Kerala or Gujarat Lions only played a season or two and don't give the model enough signal). Predictions also only start from over 5, since the score before that is too volatile to be useful.

A plain linear regression gets an R² of about 0.67. Comparing a few models with cross-validation and tuning XGBoost's hyperparameters (depth, learning rate, number of trees) with a grid search pushes that up to an R² of about 0.92.

## Project structure

```
ipl_score_deployment/
├── templates/
│   └── index.html               # prediction form
├── app.py                       # Flask app that loads the model and serves the form
├── train_model.py                # trains the model from ipl.csv, saves model.pkl + model_columns.json
├── ipl_score_prediction.ipynb    # original notebook the model was developed in
├── ipl.csv                       # training data
├── model.pkl                     # trained model
├── model_columns.json            # column order the model expects at inference time
├── style.css
├── requirements.txt
├── Deployment Link.txt           # live PythonAnywhere URL
└── README.md
```

## Running it locally

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Put `ipl.csv` in this folder and train the model:
   ```
   python train_model.py
   ```
   This creates `model.pkl` and `model_columns.json`.
3. Start the app:
   ```
   python app.py
   ```
4. Open `http://127.0.0.1:5000` and fill in the match state to get a predicted score range.


## Tech stack

Python, pandas, scikit-learn, XGBoost, Flask

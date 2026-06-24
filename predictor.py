import pandas as pd
import joblib

def predict(input_df):
    model = joblib.load('best_model.pkl')
    prediction = model.predict(input_df)
    return prediction[0]
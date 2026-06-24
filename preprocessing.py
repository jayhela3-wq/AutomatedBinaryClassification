import pandas as pd

def preprocess_data(df, target_column):

    x = df.drop(columns=[target_column])
    y = df[target_column]

    numerical_cols = x.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = x.select_dtypes(include=['object']).columns.tolist()

    feature_info = {

        'numerical_cols' : numerical_cols,
        'categorical_cols': categorical_cols
    }

    return x, y, feature_info

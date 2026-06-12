import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 

def preprocess_data(df, target_column):
    x = df.drop(columns=[target_column])
    y = df[target_column]

    numerical_cols = x.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = x.select_dtypes(include=['object']).columns

    num_imputer = SimpleImputer(strategy = 'median')
    x[numerical_cols] = num_imputer.fit_transform(x[numerical_cols])

    cat_imputer = SimpleImputer(strategy = 'most_frequent')
    x[categorical_cols] = cat_imputer.fit_transform(x[categorical_cols])

    still_missing = x.isnull().sum().sum()

    print("Remaining missing values:", still_missing)


    for col in categorical_cols:
        encoder = LabelEncoder()
        x[col] = encoder.fit_transform(x[col])

    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)

    
    scaler = StandardScaler()
    x[numerical_cols] = scaler.fit_transform(x[numerical_cols])

    return x,y
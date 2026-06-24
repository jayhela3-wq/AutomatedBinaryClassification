from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import joblib


def train(x, y,feature_info):

    numerical_cols = feature_info[numerical_cols]
    categorical_cols = feature_info[categorical_cols]

    numeric_pipe = Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler',StandardScaler)
    ])
    categorical_pipe = Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num',numeric_pipe,numerical_cols),
        ('categ',categorical_pipe,categorical_cols)
    ])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    models = {
        'Logistic Regression' : LogisticRegression(max_iter=1000),
        'Decision Tree' : DecisionTreeClassifier(random_state=42),
        'Random Forest' : RandomForestClassifier(random_state=42),
        'KNN' : KNeighborsClassifier(),
        'SVM' : SVC()
    }

    results = []
    best_model = None
    best_f1 = 0

    for name, classifier in models.items():

        pipeline = Pipeline([
            ('preprocessor',preprocessor),
            ('classifier',classifier)
        ])

        pipeline.fit(x_train, y_train)

        predictions = pipeline.predict(x_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        results.append({
            'Model':name,
            'Accuracy':round(accuracy,2),
            'Precision':round(precision,2),
            'Recall':round(recall,2),
            'F1_Score' : round(f1,2)
        })

        if f1 > best_f1 :
            best_f1 = f1
            best_model = pipeline

    
    results_df = pd.DataFrame(results)

    joblib.dump(best_model,"best_model.pkl")
    joblib.dump("feature_info","feature_info.pkl")

    return results_df




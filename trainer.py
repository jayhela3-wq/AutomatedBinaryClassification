from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import joblib


def train(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)

    models = {
        "Logistic Regression" : LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=2),
        "Random Forest" : RandomForestClassifier(random_state=2),
        "KNN" : KNeighborsClassifier(),
        "SVM" : SVC(random_state=2)
    }

    results = []

    best_model = None
    best_f1 = 0

    for name, model in models.items():
        model.fit(x_train,y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(y_test, predictions, zero_division = 0)

        recall = recall_score(y_test, predictions, zero_division=0)

        f1 = f1_score(y_test, predictions, zero_division=0)

        results.append({
            'Model' : name,
            "Accuracy" : accuracy,
            "Precision" : precision,
            "Recall" : recall,
            "F1 Score" : f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    results_df = pd.DataFrame(results)

    joblib.dump(best_model, "best_model.pkl")
    
    return results_df, best_model



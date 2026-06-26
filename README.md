# 🤖 Automated Binary Classification System

An end-to-end Machine Learning web application built using **Python**, **Scikit-learn**, and **Streamlit** that automates the binary classification workflow. The application allows users to upload any binary classification dataset, perform automated data preprocessing, train multiple machine learning models, compare their performance, and make predictions through an interactive web interface.

---

## 🌐 Live Demo

**Streamlit App:**
https://automatedbinaryclassification-bnxcvoyvdhuvaduwhnbj2z.streamlit.app

---


# 🚀 Features

* Upload any binary classification CSV dataset
* Automatic dataset analysis
* Missing value analysis
* Feature type detection (Numerical & Categorical)
* Automatic preprocessing using Scikit-learn Pipeline
* Missing value imputation
* One-Hot Encoding for categorical features
* Feature Scaling using StandardScaler
* Train multiple Machine Learning models
* Automatic best model selection
* Interactive prediction interface
* Dynamic input generation based on uploaded dataset
* Modern and user-friendly Streamlit interface

---

# 🤖 Machine Learning Models

The application trains and compares the following classification algorithms:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)

The best-performing model is automatically selected based on the **F1 Score**.

---

# 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib

---

# 📂 Project Structure

```text
AutoBinaryClassifier/
│
├── app.py
├── analysis.py
├── preprocessing.py
├── trainer.py
├── predictor.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/jayhela3-wq/AutomatedBinaryClassification.git
```

### Navigate to the project directory

```bash
cd AutomatedBinaryClassification
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the virtual environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

# 📊 Workflow

1. Upload a CSV dataset.
2. Select the target column.
3. Analyze the uploaded dataset.
4. Automatically preprocess the features.
5. Train multiple machine learning models.
6. Compare model performance.
7. Automatically select the best-performing model.
8. Enter feature values using the prediction interface.
9. View the predicted output.

---

# 📈 Performance Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

---


# 👨‍💻 Author

**Joyprakash Hela**



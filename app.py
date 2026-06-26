import streamlit as st
import pandas as pd
from analysis import analyze_dataset
from preprocessing import preprocess_data
from trainer import train
from predictor import predict
import joblib

st.set_page_config(
    page_title = 'Auto Binary Classifier',
    layout = 'wide'
)

st.title('Automated Binary Classifier')

uploaded_file = st.file_uploader('Upload your CSV dataset',type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success('Dataset uploaded successfully!')

    st.subheader('Dataset Preview')
    st.dataframe(df.head())

    st.subheader('Select Target Column')

    target_column = st.selectbox(
        'Choose the target column',
        df.columns
    )

    st.write('Selected Target:', target_column)

    analyze_dataset(df, target_column)

    x, y, feature_info = preprocess_data(df, target_column)

    st.subheader("Feature Matrix")
    st.dataframe(x.head())

    st.subheader('Target Matrix')
    st.dataframe(y.head())

    st.subheader('Preprocessed Features')
    st.dataframe(x.head())
    st.write('Remaining missing values:', x.isnull().sum().sum())

    results_df = train(x, y, feature_info)

    st.subheader('Model Comparison')
    st.dataframe(results_df)

    feature_names = joblib.load("feature_info.pkl")

    st.header("Make Prediction")

    user_input = {}

    for col in feature_info['numerical_cols']:
        user_input[col] = st.number_input(col)

    for col in feature_info['categorical_cols']:
        user_input[col] = st.select_box(label=col, options=feature_info['category_info'][col])

    input_df = pd.DataFrame([user_input])

    if st.button('Predict'):
        prediction = predict(input_df)
        st.success(f"Prediction: {prediction}")

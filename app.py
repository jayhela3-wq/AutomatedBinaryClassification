import streamlit as st
import pandas as pd

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

    st.subheader('Dataset Information')

    col1, col2 = st.columns(2)

    with col1:
        st.write('Rows:', df.shape[0])

    with col2:
        st.write('Columns:', df.shape[1])

    st.subheader('Column Names')

    st.write(list(df.columns))

    st.subheader('Select Target Column')

    target_column = st.selectbox(
        'Choose the target column',
        df.columns
    )

    st.write('Selected Target:', target_column)
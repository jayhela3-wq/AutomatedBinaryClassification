import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def analyze_dataset(df, target_column):
    if target_column is None:
        st.warning('Please select a target column')
        return

    st.header('Dataset Analysis')

    st.subheader('Dataset Overview')

    rows = df.shape[0]
    columns = df.shape[1]

    duplicates = df.duplicated().sum()

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric('Rows:', rows)

    with col2:
        st.metric('Columns:',columns)

    with col3:
        st.metric('Duplicates:', duplicates)

    

    st.subheader('Missing Values')
    
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values/len(df)*100)

    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Values': missing_values.values,
        'Missing%':missing_percentage.values
    })
    
    missing_df = missing_df.sort_values(
        by='Missing Values',
        ascending= False
    )
    st.dataframe(missing_df)

    st.subheader('Column Information')

    column_info = pd.DataFrame({
        'Column':df.columns,
        'Data Type':df.dtypes.astype(str)
    })

    st.dataframe(column_info)

    st.subheader('Target Analysis')

    unique_classes = df[target_column].nunique()

    st.write('Number of Classes:', unique_classes)

    class_counts = df[target_column].value_counts()

    st.dataframe(
        class_counts.reset_index().rename(
            columns = {
                "index": "Class",
                target_column: "Count"
            }
        )
    )

    if unique_classes == 2:
        st.success('Binary Classification Detected')

    else:
        st.warning('Multiclass Classification Detected')

    st.subheader('Class Distribution')

    class_counts = df[target_column].value_counts()

    fig, ax = plt.subplots()

    ax.bar(
        class_counts.index.astype(str),
        class_counts.values
    )

    ax.set_title('Target Class Distribution')
    st.pyplot(fig)

    st.subheader('Statistical Summary')
    st.dataframe(df.describe())




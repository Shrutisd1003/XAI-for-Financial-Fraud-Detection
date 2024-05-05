import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def visualize_data(self):
        dataset = self.data
        dataset["Prediction"] = dataset["Prediction"].map({"Fraud":1, "Not Fraud":0})
        df_type = st.selectbox('Choose a dataset', ['All transactions', 'Only Fraudulent Transactions', 'Only Non Fraudulent Transactions'])
        if df_type == "All transactions":
            df = dataset
        elif df_type == "Only Fraudulent Transactions":
            df = dataset[dataset["Prediction"] == 1]
        elif df_type == "Only Non Fraudulent Transactions":
            df = dataset[dataset["Prediction"] == 0]

        plot_type = st.selectbox('Choose a type of plot', ['Histogram', 'Box Plot', 'Pie Chart', 'Scatter Plot', 'Heatmap'])
        
        if plot_type == 'Histogram':
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if numeric_columns.empty:
                st.warning('No numeric columns in the data to visualize.')
            else:
                column_to_visualize = st.selectbox('Choose a column to visualize', numeric_columns)
                fig, ax = plt.subplots()
                ax.hist(df[column_to_visualize])
                ax.set_title(f'Histogram of {column_to_visualize}')
                ax.set_xlabel(column_to_visualize)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
        
        elif plot_type == 'Box Plot':
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if numeric_columns.empty:
                st.warning('No numeric columns in the data to visualize.')
            else:
                column_to_visualize = st.selectbox('Choose a column to visualize', numeric_columns)
                fig, ax = plt.subplots()
                ax.boxplot(df[column_to_visualize].dropna())
                ax.set_title(f'Box Plot of {column_to_visualize}')
                ax.set_ylabel(column_to_visualize)
                st.pyplot(fig)
        
        elif plot_type == 'Pie Chart':
            nonnumeric_columns = df.select_dtypes(include=['object']).columns
            if nonnumeric_columns.empty:
                st.warning('No non numeric columns in the data to visualize.')
            else:
                column_to_visualize = st.selectbox('Choose a column to visualize', nonnumeric_columns)
                fig, ax = plt.subplots()
                df[column_to_visualize].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', textprops={'fontsize': 'small'})
                ax.set_title(f'Pie Chart of {column_to_visualize}')
                ax.set_ylabel('')
                st.pyplot(fig)
        
        elif plot_type == 'Scatter Plot':
            left, right = st.columns(2)
            with left:
                x_col = st.selectbox('Choose values on X axis', df.select_dtypes(include=[np.number]).columns)
            with right:
                y_col = st.selectbox('Choose values on Y axis', df.select_dtypes(include=[np.number]).columns)
            if x_col == y_col:
                st.warning('Please select two different columns for scatter plot.')
            else:
                fig, ax = plt.subplots()
                ax.scatter(df[x_col], df[y_col])
                ax.set_title(f'Scatter Plot of {x_col} vs {y_col}')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                st.pyplot(fig)
        
        elif plot_type == 'Heatmap':
            numeric_data = df.select_dtypes(include=[np.number])
            corr = numeric_data.corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, ax=ax)
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)
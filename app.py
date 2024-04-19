import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
from Modules.data_cleaning import cleaned_data
from Modules.llm_explainer import generate_response

def load_model():
    with open('Model building/skf_random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def main():
    data = pd.DataFrame()
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if st.button("Predict"):
        data = pd.read_csv(uploaded_file)
        data = cleaned_data(data)

        model = load_model()
        prediction = model.predict(data).round().astype(int)

        counts = np.bincount(prediction)
        fig, ax = plt.subplots()
        ax.pie(counts, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        transaction_ids_list = data.index.tolist()
        transaction_id = st.selectbox("Transaction IDs", transaction_ids_list)
        if st.button("Explain"):
            response = generate_respone(data, transaction_id, prediction[transaction_ids_list.index(transaction_id)])
            st.write(response)

if __name__ == "__main__":
    main()
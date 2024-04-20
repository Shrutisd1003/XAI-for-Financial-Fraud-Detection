import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
from Modules.data_cleaning import cleaned_data
from Modules.llm_explainer import generate_response

def load_model():
    with open('Model building/prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_and_clean_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    transaction_ids, modified_data = cleaned_data(data)
    return transaction_ids, modified_data

def main():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        transaction_ids, data = load_and_clean_data(uploaded_file)
        model = load_model()
        prediction = model.predict(data).round().astype(int)

        transaction_ids = transaction_ids.tolist()
        transaction_id = st.selectbox("Transaction IDs", transaction_ids)

            # if transaction_id != "-Select-":
            #     print("explain button displayed")
        idx = transaction_ids.index(transaction_id)
            
        if st.button("Explain"):
            try:
                response = generate_response(model, data, idx, prediction[idx])
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.write("Select a file")

if __name__ == "__main__":
    main()

# TO DO
# 1. print labels instead of encodings
# 2. display pie chart
# 3. diplay transaction ids category wise
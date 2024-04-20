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

@st.cache_data
def load_and_clean_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    transaction_ids, modified_data = cleaned_data(data)
    return transaction_ids, modified_data

def main():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    print("file uploader")
    if uploaded_file:
        transaction_ids, data = load_and_clean_data(uploaded_file)
        print("data cleaned")

        if st.button("Predict"):
            print("predict clicked")
            model = load_model()
            prediction = model.predict(data).round().astype(int)
            print("predictions made")

            transaction_ids = transaction_ids.tolist()
            transaction_id = st.selectbox("Transaction IDs", transaction_ids)
            print(f"transaction id - {transaction_id}")

            # if transaction_id != "-Select-":
            #     print("explain button displayed")
            idx = transaction_ids.index(transaction_id)
            if st.button("Explain"):
                print("explain button clicked")
                response = generate_response(model, data, idx, prediction[idx])
                print("response generated")
                st.write(response)
    else:
        st.write("Select a file")
if __name__ == "__main__":
    main()

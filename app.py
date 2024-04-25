import time
import pickle
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
from Modules.data_cleaning import cleaned_data
from Modules.llm_explainer import generate_response
from Modules.data_visualizer import DataVisualizer

def load_model():
    with open('Model building/prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_and_clean_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    data.to_csv("original_data.csv", index=False)
    transaction_ids, modified_data = cleaned_data(data)
    return transaction_ids, modified_data

def color_coding(row):
    return ['background-color: red' if row["Prediction"] == 'Fraud' else '' for _ in row]

def stream_data(words):
    for word in words.split(" "):
        yield word + " "
        time.sleep(0.02)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- LOAD ASSETS---
lottie_transaction = load_lottieurl("https://lottie.host/98295178-0a8f-4162-9a00-265d3f498cfd/SL87hpUcLF.json")

def main():
    st.set_page_config(layout="wide")
    left_padding, middle, right_padding = st.columns([1,4,1])
    with left_padding:
        pass
    with middle:
        left, right = st.columns([2,1])
        with left:
            st.title("XAI for Financial Fraud Detection")
        with right:
            st_lottie(lottie_transaction, height=200, width=200, key="cash transaction")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            transaction_ids, modified_data = load_and_clean_data(uploaded_file)
            model = load_model()
            prediction = model.predict(modified_data).round().astype(int)
            
            original_data = pd.read_csv("original_data.csv")

            st.header("Results")
            original_data["Prediction"] = ["Fraud" if x == 1 else "Not Fraud" for x in prediction]
            st.dataframe(original_data.style.apply(color_coding, axis=1), hide_index=True)
            fraud_percent = (np.sum(prediction) / len(prediction)) * 100
            st.write(f"{fraud_percent} % of the transactions were fraudulent")

            st.header("Visualizations")
            data_visualizer = DataVisualizer(original_data)
            data_visualizer.visualize_data()
            
            st.header("Explainable AI")
            transaction_ids = transaction_ids.tolist()
            transaction_ids.insert(0, "-Select-")
            transaction_id = st.selectbox("Transaction IDs", transaction_ids)

            if transaction_id != "-Select-":
                idx = transaction_ids.index(transaction_id) - 1
                if st.button("Explain"):
                    try:
                        with st.spinner('Analyzing...'):
                            response = generate_response(model, modified_data, idx, prediction[idx])
                            st.write_stream(stream_data(response))
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
    with right_padding:
        pass

if __name__ == "__main__":
    main()
